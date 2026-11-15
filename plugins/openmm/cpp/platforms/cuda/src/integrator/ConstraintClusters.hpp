// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Sorting a System's distance constraints into the two kinds of group that the
// Langevin middle integrator's GPU solvers handle:
//
//  - rigid three-site water, solved in closed form (Miyamoto and Kollman 1992):
//    three atoms joined pairwise by constraints and by nothing else, two of the
//    three lengths equal;
//  - stars of one central atom and one to three outer atoms, solved iteratively
//    (Ryckaert, Ciccotti and Berendsen 1977): every outer atom is constrained to
//    the central atom only, and all outer atoms share one length and one mass.
//
// Constraints that fit neither shape would need a coupled solver, which the
// plugin does not have; they are only counted.
//
// The grouping, the order of the groups and the order of atoms within a group
// are fixed: the solvers' rounding depends on atom order, and the device arrays
// depend on group order.

#pragma once

#include <cmath>
#include <map>
#include <vector>

#include "openmm/System.h"
#include "openmm/common/ComputeVectorTypes.h"

namespace NvOpenMM {

struct ConstraintClusters {
    std::vector<OpenMM::mm_int4>   settleAtoms;    // (oxygen-like atom, other, other, 0)
    std::vector<OpenMM::mm_float2> settleParams;   // (length to the oxygen-like atom, length between the others)
    std::vector<OpenMM::mm_int4>   shakeAtoms;     // (central, outer 1, outer 2 or -1, outer 3 or -1)
    std::vector<OpenMM::mm_float4> shakeParams;    // (central 1/m, 0.5/(central 1/m + outer 1/m), length^2, outer 1/m)
    int numLeftoverConstraints = 0;
};

namespace constraint_clusters_detail {

// One constraint as the System lists it: endpoints in the System's order.
struct ConstraintEdge {
    int first;
    int second;
    double length;
};

// A star under construction, keyed by its central atom.
struct StarCandidate {
    bool usable = true;
    double centralInverseMass = 0.0;
    std::vector<int> outerAtoms;          // in the order they were accepted
    double lastLength = 0.0;              // length of the most recently accepted outer atom
    double lastOuterInverseMass = 0.0;    // 1/m of the most recently accepted outer atom
};

// Relative difference test used to decide whether a new outer atom matches the
// star.  Written as "greater than" so that a NaN (from infinity minus infinity,
// i.e. two massless outer atoms) counts as a match.
inline bool differsRelatively(double value, double reference) {
    return std::fabs(value - reference) / reference > 1e-8;
}

}  // namespace constraint_clusters_detail

inline ConstraintClusters findConstraintClusters(const OpenMM::System& system) {
    using namespace constraint_clusters_detail;
    ConstraintClusters result;

    const int numAtoms = system.getNumParticles();
    const int numConstraints = system.getNumConstraints();
    if (numAtoms <= 0 || numConstraints <= 0)
        return result;

    // Step 1: keep every constraint that has at least one atom with mass.  A
    // constraint between two massless atoms has no effect on the dynamics and is
    // ignored entirely (it is not even counted as unsolved).
    std::vector<ConstraintEdge> edges;
    edges.reserve(numConstraints);
    for (int i = 0; i < numConstraints; i++) {
        ConstraintEdge e;
        system.getConstraintParameters(i, e.first, e.second, e.length);
        if (system.getParticleMass(e.first) == 0.0 && system.getParticleMass(e.second) == 0.0)
            continue;
        edges.push_back(e);
    }

    // How many kept constraints touch each atom, and which edges they are.  A
    // pair constrained twice counts twice.
    std::vector<int> degree(numAtoms, 0);
    std::vector<std::vector<int>> edgesOfAtom(numAtoms);
    for (int k = 0; k < (int) edges.size(); k++) {
        degree[edges[k].first]++;
        degree[edges[k].second]++;
        edgesOfAtom[edges[k].first].push_back(k);
        edgesOfAtom[edges[k].second].push_back(k);
    }

    // Atoms that have been placed in an accepted group.
    std::vector<char> taken(numAtoms, 0);

    // Step 2: rigid triangles.  An atom with exactly two constraints whose two
    // partners also have exactly two constraints each and are constrained to
    // each other forms an isolated triangle.  Scanning atoms in increasing index
    // and requiring the scanned atom to be the lowest of the three visits each
    // triangle once, in increasing order of its lowest index.
    auto otherEnd = [&](int edge, int atom) {
        return edges[edge].first == atom ? edges[edge].second : edges[edge].first;
    };
    for (int p = 0; p < numAtoms; p++) {
        if (degree[p] != 2)
            continue;
        const int e1 = edgesOfAtom[p][0];
        const int e2 = edgesOfAtom[p][1];
        int q = otherEnd(e1, p);
        int r = otherEnd(e2, p);
        double lengthPQ = edges[e1].length;
        double lengthPR = edges[e2].length;
        if (q == r || q == p || r == p || q < p || r < p)
            continue;
        if (degree[q] != 2 || degree[r] != 2)
            continue;
        // Find the edge joining q and r among q's two edges.
        int edgeQR = -1;
        for (int e : edgesOfAtom[q])
            if (e != e1 && otherEnd(e, q) == r)
                edgeQR = e;
        if (edgeQR < 0)
            continue;
        if (q > r) {
            std::swap(q, r);
            std::swap(lengthPQ, lengthPR);
        }
        // Compare the lengths as float: the solver works with float lengths, so
        // two lengths that round to the same float are equal for it.  The atom
        // whose two constraints are equal comes first (the oxygen of a water).
        const float fPQ = (float) lengthPQ;
        const float fPR = (float) lengthPR;
        const float fQR = (float) edges[edgeQR].length;
        OpenMM::mm_int4 atoms(0, 0, 0, 0);
        OpenMM::mm_float2 params(0.0f, 0.0f);
        if (fPQ == fPR) {
            atoms = OpenMM::mm_int4(p, q, r, 0);
            params = OpenMM::mm_float2(fPQ, fQR);
        } else if (fPQ == fQR) {
            atoms = OpenMM::mm_int4(q, p, r, 0);
            params = OpenMM::mm_float2(fPQ, fPR);
        } else if (fPR == fQR) {
            atoms = OpenMM::mm_int4(r, p, q, 0);
            params = OpenMM::mm_float2(fPR, fPQ);
        } else {
            continue;   // three different lengths: left to the star search
        }
        result.settleAtoms.push_back(atoms);
        result.settleParams.push_back(params);
        taken[p] = taken[q] = taken[r] = 1;
    }

    // Step 3: stars.  Walk the remaining constraints in System order, growing one
    // candidate star per central atom.  Atoms involved in a shape that cannot be
    // a star (a chain, a ring, an outer atom with other constraints) are marked
    // "spoiled", and spoiling spreads to every star reached through them.
    std::map<int, StarCandidate> stars;   // ordered by central atom
    std::vector<char> spoiled(numAtoms, 0);

    // Give up on a star: make it unusable, spoil its central atom and every
    // outer atom it holds, then give up in the same way on each still-usable
    // star centred on one of those outer atoms, and so on.  This applies even if
    // the first star was already unusable.  A work list of central atoms replaces
    // recursion so that a long chain cannot overflow the call stack; the set of
    // stars reached does not depend on the visiting order, because a star is
    // entered only while usable and becomes unusable when entered.
    auto abandonStar = [&](int central) {
        stars[central].usable = false;
        std::vector<int> pending{central};
        while (!pending.empty()) {
            const int key = pending.back();
            pending.pop_back();
            const StarCandidate& star = stars[key];
            spoiled[key] = 1;
            for (int atom : star.outerAtoms)
                spoiled[atom] = 1;
            for (int atom : star.outerAtoms) {
                auto it = stars.find(atom);
                if (it != stars.end() && it->second.usable) {
                    it->second.usable = false;
                    pending.push_back(atom);
                }
            }
        }
    };

    for (const ConstraintEdge& e : edges) {
        const int a = e.first;
        const int b = e.second;
        // A constraint whose first endpoint already belongs to a triangle is
        // handled.  Only the first endpoint is looked at.
        if (taken[a])
            continue;

        // The central atom is the endpoint with more than one constraint (the
        // first endpoint wins a tie); if neither has, the lower index.
        int central, outer;
        if (degree[a] > 1) {
            central = a; outer = b;
        } else if (degree[b] > 1) {
            central = b; outer = a;
        } else if (a < b) {
            central = a; outer = b;
        } else {
            central = b; outer = a;
        }

        auto found = stars.find(central);
        if (found == stars.end()) {
            StarCandidate fresh;
            fresh.centralInverseMass = 1.0 / system.getParticleMass(central);
            found = stars.emplace(central, fresh).first;
        }
        StarCandidate& star = found->second;

        // Accept the outer atom if the star has room and the atom matches the
        // length and mass of the previously accepted one.  A mismatch makes the
        // star unusable but spoils no atom by itself.
        const double outerInverseMass = 1.0 / system.getParticleMass(outer);
        const bool full = star.outerAtoms.size() >= 3;
        const bool mismatch = !star.outerAtoms.empty() &&
                              (differsRelatively(e.length, star.lastLength) ||
                               differsRelatively(outerInverseMass, star.lastOuterInverseMass));
        if (full || mismatch) {
            star.usable = false;
        } else {
            star.outerAtoms.push_back(outer);
            star.lastLength = e.length;
            star.lastOuterInverseMass = outerInverseMass;
        }

        // An outer atom must have no constraint besides this one, and neither
        // endpoint may already be spoiled.  Otherwise this star is abandoned, and
        // so is a star centred on the outer atom (the case of a chain).
        if (degree[outer] != 1 || spoiled[a] || spoiled[b]) {
            abandonStar(central);
            auto other = stars.find(outer);
            if (other != stars.end() && other->second.usable)
                abandonStar(outer);
        }
    }

    // Step 4: keep the stars that survived, in increasing order of central atom:
    // still usable, no spoiled atom, and every constraint of the central atom
    // accounted for by an outer atom.
    for (const auto& entry : stars) {
        const int central = entry.first;
        const StarCandidate& star = entry.second;
        if (!star.usable || spoiled[central])
            continue;
        if ((int) star.outerAtoms.size() != degree[central])
            continue;
        bool clean = true;
        for (int atom : star.outerAtoms)
            if (spoiled[atom])
                clean = false;
        if (!clean)
            continue;

        const int n = (int) star.outerAtoms.size();
        result.shakeAtoms.push_back(OpenMM::mm_int4(central,
                                                    star.outerAtoms[0],
                                                    n > 1 ? star.outerAtoms[1] : -1,
                                                    n > 2 ? star.outerAtoms[2] : -1));
        const double w0 = star.centralInverseMass;
        const double w1 = star.lastOuterInverseMass;
        result.shakeParams.push_back(OpenMM::mm_float4((float) w0,
                                                       (float) (0.5 / (w0 + w1)),
                                                       (float) (star.lastLength * star.lastLength),
                                                       (float) w1));
        taken[central] = 1;
        for (int atom : star.outerAtoms)
            taken[atom] = 1;
    }

    // Step 5: a kept constraint whose first endpoint is in no group is unsolved.
    for (const ConstraintEdge& e : edges)
        if (!taken[e.first])
            result.numLeftoverConstraints++;

    return result;
}

}  // namespace NvOpenMM
