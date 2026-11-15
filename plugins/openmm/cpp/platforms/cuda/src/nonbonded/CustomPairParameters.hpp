// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "openmm/CustomNonbondedForce.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "CudaContext.h"
#include "cupo/expression.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace NvOpenMM {
class NonbondedForce;
class InteractionModel;

// OpenMM definitions, parameter ordering and exclusions. Libcupo's expression
// instance owns compilation, parameter preparation and stable evaluator buffers;
// the common nonbonded kernel owns geometry, streams, graphs and output.
class CustomPairParameters {
public:
    explicit CustomPairParameters(OpenMM::CudaContext& cu);
    ~CustomPairParameters();
    void initialize(const OpenMM::System& system, const NonbondedForce& force,
                    InteractionModel& model);
    // Effective native records in System order; lattice weights come from the model.
    void setManagedParameters(const std::vector<float>& q_sigma_epsilon);
    // OpenMM notifies the common owner after changing its atom permutation.
    // Defer comparison and uploads until the next preparation outside capture.
    void invalidateAtomOrder() { atom_order_dirty_ = true; }
    void validateDefinition(const NonbondedForce& force) const;
    // Allocate stable buffers lazily and complete the initial topology upload
    // before DSpace creation reads exclusions. Called once, outside capture.
    void setup();
    // The descriptor and prepared records borrow storage from the expression
    // instance. Their addresses remain stable until this helper is destroyed.
    cupo_pair_desc pairDescriptor() const;
    const void* particles() const { return binding_.d_particles; }
    size_t particleStride() const { return binding_.particle_stride_bytes; }
    int exclusionCount() const { return static_cast<int>(exclusions_.size()/2); }
    const int* exclusions() const { return d_exclusions_; }

    // Enqueue updates/preprocessing on OpenMM's current stream, outside capture.
    // The owner must join any prior side-stream graph readers onto that stream
    // before calling either update method. Upload events protect host staging;
    // they do not by themselves fence readers in another stream.
    // native_parameters borrows current CUDA-order q/sigma/epsilon records for
    // fused Coulomb; standalone Custom passes nullptr.
    void prepareStep(OpenMM::ContextImpl& context, cudaStream_t stream,
                     const float* native_parameters);
    // The common owner must first call validateDefinition before committing any
    // native parameters or tails. This applies that same validated definition;
    // numeric staging is still checked before committing host/device records.
    void copyParametersToContext(OpenMM::ContextImpl& context,
                                 const NonbondedForce& force);
    // Destroy all dependent graphs and pair workspaces before this helper.

private:
    void uploadParameters(const OpenMM::CustomNonbondedForce& force);
    void refreshStandaloneWeights(const std::map<std::string,double>& globals);
    void refreshManagedColumns();
    void reorderParameters();
    void waitForUploads();
    void upload(void* destination, const void* source, size_t bytes, cudaStream_t stream);
    OpenMM::CudaContext& cu_;
    InteractionModel* model_ = nullptr;
    std::unique_ptr<OpenMM::CustomNonbondedForce> definition_;
    cupo_expression_program_handle program_ = nullptr;
    cupo_expression_instance_handle instance_ = nullptr;
    cupo_expression_binding binding_{};
    int* d_exclusions_ = nullptr;
    cudaEvent_t uploads_done_ = nullptr;
    bool allocated_ = false, uploads_pending_ = false;
    bool host_parameters_dirty_ = true, atom_order_dirty_ = true;
    bool managed_charges_dirty_ = false;
    bool records_pending_ = false;
    std::vector<float> parameters_, reordered_, globals_;
    std::vector<std::string> particle_names_, global_names_;
    std::map<std::string,double> weight_globals_;
    std::vector<int> exclusions_, reordered_exclusions_, atom_order_;
    int particle_count_ = 0, parameter_count_ = 0;
    int public_parameter_count_ = 0, charge_column_ = -1, dispersion_column_ = -1;
    bool standard_ = false, generated_lj_ = false, native_charge_column_ = false;
    bool weight_records_dirty_ = true;
    bool managed_charges_ = false;
    bool managed_dispersion_ = false;
    bool trace_updates_ = false;
    unsigned reorder_count_ = 0;
    unsigned record_upload_count_ = 0, exclusion_upload_count_ = 0;
};

}  // namespace NvOpenMM
