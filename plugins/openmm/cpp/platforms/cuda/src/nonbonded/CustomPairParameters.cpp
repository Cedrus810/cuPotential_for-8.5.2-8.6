// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "CustomPairParameters.hpp"
#include "InteractionModel.hpp"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/OpenMMException.h"
#include "lepton/Parser.h"
#include "lepton/ParsedExpression.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>

namespace NvOpenMM {
namespace {
void expressionCheck(cupo_status status) {
    if (status != CUPO_SUCCESS)
        throw OpenMM::OpenMMException(std::string("CustomNonbondedForce expression: ") +
                                      cupo_last_error());
}
void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw OpenMM::OpenMMException(std::string("CustomNonbondedForce CUDA: ") +
                                      cudaGetErrorString(status));
}
class CustomForceInfo : public OpenMM::ComputeForceInfo {
public:
    explicit CustomForceInfo(const OpenMM::CustomNonbondedForce& force) : force_(force) {}
    int getNumParticleGroups() override { return force_.getNumExclusions(); }
    void getParticlesInGroup(int index, std::vector<int>& particles) override {
        int first,second;
        force_.getExclusionParticles(index,first,second);
        particles={first,second};
    }
    bool areGroupsIdentical(int, int) override { return true; }
    bool areParticlesIdentical(int first, int second) override {
        std::vector<double> a,b;
        force_.getParticleParameters(first,a);
        force_.getParticleParameters(second,b);
        return a==b;
    }
private:
    const OpenMM::CustomNonbondedForce& force_;
};

std::unique_ptr<OpenMM::CustomNonbondedForce> expressionDefinition(const NonbondedForce& force) {
    if (force.hasCutoffExpression())
        return std::make_unique<OpenMM::CustomNonbondedForce>(force.getCutoffExpressionDefinition());
    // Unusual lattice configurations retain the same native LJ model through a
    // generated evaluator. These private records receive effective sigma/epsilon
    // after native offsets; the native analytical tail remains with the owner.
    const auto field = [&](std::string name) {
        bool collision;
        do {
            collision=false;
            for (int i=0;i<force.getNumGlobalParameters();++i) {
                const auto& global=force.getGlobalParameterName(i);
                collision=collision || global==name || global==name+"1" || global==name+"2";
            }
            if (collision) name+="_";
        } while (collision);
        return name;
    };
    const std::string sigma=field("__cupo_native_sigma"), epsilon=field("__cupo_native_epsilon");
    const std::string mixed=force.getLJCombiningRule()==NonbondedForce::Geometric ?
        "sqrt("+sigma+"1*"+sigma+"2)" : "(0.5*("+sigma+"1+"+sigma+"2))";
    const std::string ratio="("+mixed+"/r)";
    auto result=std::make_unique<OpenMM::CustomNonbondedForce>(
        "4*sqrt("+epsilon+"1*"+epsilon+"2)*("+ratio+"^12-"+ratio+"^6)");
    result->setNonbondedMethod(OpenMM::CustomNonbondedForce::CutoffPeriodic);
    result->setCutoffDistance(force.getCutoffDistance());
    result->setUseSwitchingFunction(force.getUseSwitchingFunction());
    result->setSwitchingDistance(force.getSwitchingDistance());
    result->addPerParticleParameter(sigma);
    result->addPerParticleParameter(epsilon);
    for (int i=0;i<force.getNumGlobalParameters();++i)
        result->addGlobalParameter(force.getGlobalParameterName(i),force.getGlobalParameterDefaultValue(i));
    for (int i=0;i<force.getNumParticles();++i) {
        double q,s,e;
        force.getParticleParameters(i,q,s,e);
        result->addParticle({s,e});
    }
    for (int i=0;i<force.getNumExceptions();++i) {
        int first,second;
        double q,s,e;
        force.getExceptionParameters(i,first,second,q,s,e);
        result->addExclusion(first,second);
    }
    return result;
}
}

CustomPairParameters::CustomPairParameters(OpenMM::CudaContext& cu) : cu_(cu) {}

CustomPairParameters::~CustomPairParameters() {
    OpenMM::ContextSelector selector(cu_);
    // The owner drains/destroys its graphs and pair workspaces first. This wait
    // also completes our host uploads and preprocessing on OpenMM's stream.
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cu_.getCurrentStream()));
    if (uploads_done_) cudaEventDestroy(uploads_done_);
    cupo_expression_instance_destroy(instance_);
    cupo_expression_program_destroy(program_);
    if (d_exclusions_) cudaFree(d_exclusions_);
}

void CustomPairParameters::initialize(const OpenMM::System&,
                                const NonbondedForce& owner, InteractionModel& model) {
    OpenMM::ContextSelector selector(cu_);
    if (cu_.getUseDoublePrecision())
        throw OpenMM::OpenMMException("CustomNonbondedForce supports single or mixed precision");
    if (cu_.getNumContexts()!=1)
        throw OpenMM::OpenMMException("CustomNonbondedForce currently supports one CUDA device per Context");
    cudaCheck(cudaEventCreateWithFlags(&uploads_done_,cudaEventDisableTiming));
    model_=&model;
    standard_=model.hasBuiltinParameters();
    generated_lj_=standard_ && !owner.hasCutoffExpression();
    definition_=expressionDefinition(owner);
    const auto& force=*definition_;
    const char* trace=std::getenv("CUPO_PLUGIN_GRAPH_TRACE");
    trace_updates_=trace && trace[0]=='1';
    particle_count_=force.getNumParticles();
    parameter_count_=force.getNumPerParticleParameters();
    public_parameter_count_=parameter_count_;
    for (int i=0;i<parameter_count_;++i)
        particle_names_.push_back(force.getPerParticleParameterName(i));
    for (int i=0;i<force.getNumGlobalParameters();++i)
        global_names_.push_back(force.getGlobalParameterName(i));
    managed_charges_=model.latticeSum.has_value();
    managed_dispersion_=model.dispersion.has_value();
    native_charge_column_=standard_ && managed_charges_ && !managed_dispersion_ &&
        model.latticeSum->definition().weightExpression=="q";
    if (!standard_) {
        if (model.latticeSum) model.latticeSum->initializeWeightSchema(particle_names_,global_names_);
        if (model.dispersion) model.dispersion->initializeWeightSchema(particle_names_,global_names_);
    }
    const auto appendHiddenField = [&](std::string name) {
        while (force.getEnergyFunction().find(name)!=std::string::npos ||
                std::find(particle_names_.begin(),particle_names_.end(),name)!=particle_names_.end() ||
                std::find(global_names_.begin(),global_names_.end(),name)!=global_names_.end() ||
                std::find(global_names_.begin(),global_names_.end(),name+"1")!=global_names_.end() ||
                std::find(global_names_.begin(),global_names_.end(),name+"2")!=global_names_.end())
            name+="_";
        ++parameter_count_;
        particle_names_.push_back(name);
    };
    if (managed_charges_) {
        charge_column_=parameter_count_;
        appendHiddenField("__cupo_managed_charge");
    }
    if (managed_dispersion_) {
        dispersion_column_=parameter_count_;
        appendHiddenField("__cupo_managed_b6");
    }
    globals_.assign(global_names_.size(),0);
    for (int i=0;i<force.getNumExclusions();++i) {
        int first,second;
        force.getExclusionParticles(i,first,second);
        exclusions_.push_back(first); exclusions_.push_back(second);
    }
    int device=0;
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDevice(&device));
    cudaCheck(cudaGetDeviceProperties(&properties,device));
    std::vector<const char*> particle_names, global_names;
    for (const auto& name:particle_names_) particle_names.push_back(name.c_str());
    for (const auto& name:global_names_) global_names.push_back(name.c_str());
    // Keep the original expression and switching settings for OpenMM updates
    // and tail integration. Libcupo owns the GPU expression's optimized form.
    cupo_expression_desc desc{};
    desc.struct_bytes=sizeof(desc);
    desc.abi_version=CUPO_EXPRESSION_DESC_VERSION;
    std::string energy=force.getEnergyFunction();
    if (standard_ && (managed_charges_ || managed_dispersion_)) {
        // Resolve the user's local definitions before adding managed terms:
        // a definition such as r=2 must not redefine managed Coulomb's distance.
        std::ostringstream reference;
        const auto expression=Lepton::Parser::parse(energy);
        reference << std::setprecision(17) << expression;
        energy=reference.str();
        if (managed_dispersion_) {
            // Shift only the full unsplit non-Coulomb expression. The compiler
            // adds the finite b6_i*b6_j*(LR6(r)-LR6(rc)) complement directly.
            std::ostringstream atCutoff;
            atCutoff << std::setprecision(17) << expression.optimize({{"r",force.getCutoffDistance()}});
            energy="("+energy+")-("+atCutoff.str()+")";
        }
    }
    if (standard_ && managed_charges_) {
        // The compiler accepts the full unsplit pair potential and subtracts
        // the plans' reciprocal complements. Match its FP32 coupling exactly
        // so it can lower this ordinary Coulomb term to the stable SR helper
        // before applying switching to the remaining cutoff expression.
        const auto& charge=particle_names_[charge_column_];
        std::ostringstream coulomb;
        coulomb << std::setprecision(std::numeric_limits<double>::max_digits10)
                << static_cast<float>(model.latticeSum->definition().kernel.coupling)
                << '*' << charge << "1*" << charge << "2/r";
        energy="("+energy+")+("+coulomb.str()+")";
    }
    desc.energy=energy.c_str();
    desc.particle_names=particle_names.data();
    desc.particle_parameter_count=static_cast<uint32_t>(particle_names.size());
    desc.global_names=global_names.data();
    desc.global_count=static_cast<uint32_t>(global_names.size());
    desc.compute_capability=properties.major*10+properties.minor;
    desc.cutoff=force.getCutoffDistance();
    desc.use_switching=force.getUseSwitchingFunction();
    desc.switch_distance=force.getSwitchingDistance();
    cupo_expression_lattice_direct_term lattice_direct[2]{};
    if (managed_charges_) {
        // This hidden input contains the authoritative effective charges also
        // passed to KSpace. Libcupo copies the plan and weight expression during
        // compilation and owns all derived-weight preparation.
        auto& term=lattice_direct[desc.lattice_direct_term_count++];
        term.plan=model.latticeSum->plan();
        term.weight_expression=particle_names_[charge_column_].c_str();
        if (!standard_) {
            // Preserve canonical Coulomb recognition in a full Custom source.
            // A simple public FP32 field has exactly the authoritative weight
            // bits; derived weights use their prepared hidden field instead.
            const auto& text=model.latticeSum->definition().weightExpression;
            const auto first=text.find_first_not_of(" \t\r\n"), last=text.find_last_not_of(" \t\r\n");
            const auto name=text.substr(first,last-first+1);
            for (int i=0;i<public_parameter_count_;++i)
                if (name==particle_names_[i])
                    term.weight_expression=particle_names_[i].c_str();
        }
    }
    if (managed_dispersion_) {
        auto& term=lattice_direct[desc.lattice_direct_term_count++];
        term.plan=model.dispersion->plan();
        term.weight_expression=particle_names_[dispersion_column_].c_str();
    }
    desc.lattice_direct_terms=lattice_direct;
    expressionCheck(cupo_expression_compile(&desc,&program_));
    uploadParameters(force);
    if (!standard_) {
        std::map<std::string,double> defaults;
        for (int i=0;i<force.getNumGlobalParameters();++i)
            defaults[force.getGlobalParameterName(i)]=force.getGlobalParameterDefaultValue(i);
        refreshStandaloneWeights(defaults);
    }
    if (!generated_lj_) cu_.addForce(new CustomForceInfo(*definition_));
}

void CustomPairParameters::waitForUploads() {
    if (uploads_pending_) {
        // The event precedes pair kernels: retain host sources until their copies
        // finish without waiting for force-only execution itself to finish.
        cudaCheck(cudaEventSynchronize(uploads_done_));
        uploads_pending_=false;
    }
}

void CustomPairParameters::upload(
        void* destination,const void* source,size_t bytes,cudaStream_t stream) {
    cudaCheck(cudaMemcpyAsync(destination,source,bytes,cudaMemcpyHostToDevice,stream));
    cudaCheck(cudaEventRecord(uploads_done_,stream));
    uploads_pending_=true;
}

void CustomPairParameters::uploadParameters(const OpenMM::CustomNonbondedForce& force) {
    std::vector<float> next(static_cast<size_t>(particle_count_)*parameter_count_);
    const int public_count=public_parameter_count_;
    std::vector<double> values;
    for (int i=0;i<particle_count_;++i) {
        force.getParticleParameters(i,values);
        if (values.size()!=static_cast<size_t>(public_count))
            throw OpenMM::OpenMMException("CustomNonbondedForce: wrong number of particle parameters");
        for (int j=0;j<public_count;++j) {
            const float value=static_cast<float>(values[j]);
            if (!std::isfinite(value))
                throw OpenMM::OpenMMException("CustomNonbondedForce parameters must be finite FP32 values");
            next[static_cast<size_t>(i)*parameter_count_+j]=value;
        }
        if (!parameters_.empty())
            for (int j=public_count;j<parameter_count_;++j)
                next[static_cast<size_t>(i)*parameter_count_+j]=parameters_[static_cast<size_t>(i)*parameter_count_+j];
    }
    if (next.size()!=parameters_.size() ||
            (!next.empty() && std::memcmp(next.data(),parameters_.data(),next.size()*sizeof(float))!=0)) {
        parameters_.swap(next);
        host_parameters_dirty_=true;
        weight_records_dirty_=true;
    }
}

void CustomPairParameters::setManagedParameters(const std::vector<float>& values) {
    if (!standard_ || values.size()!=static_cast<size_t>(particle_count_)*3)
        throw OpenMM::OpenMMException("CustomNonbondedForce: invalid managed native parameters");
    if (generated_lj_) {
        for (float value:values)
            if (!std::isfinite(value))
                throw OpenMM::OpenMMException("CustomNonbondedForce: managed native parameters must be finite");
        for (int i=0;i<particle_count_;++i) {
            for (int column=0;column<2;++column) {
                auto& previous=parameters_[static_cast<size_t>(i)*parameter_count_+column];
                const float value=values[3*i+column+1];
                if (std::memcmp(&previous,&value,sizeof(float))!=0) {
                    previous=value;
                    host_parameters_dirty_=true;
                }
            }
        }
    }
    refreshManagedColumns();
}

void CustomPairParameters::refreshManagedColumns() {
    const auto validate = [&](const std::vector<float>& weights) {
        if (weights.size()!=static_cast<size_t>(particle_count_))
            throw OpenMM::OpenMMException("CustomNonbondedForce: managed lattice weight count mismatch");
        for (float value:weights)
            if (!std::isfinite(value))
                throw OpenMM::OpenMMException("CustomNonbondedForce: managed lattice weights must be finite");
    };
    if (managed_charges_) validate(model_->latticeSum->weights());
    if (managed_dispersion_) validate(model_->dispersion->weights());
    const auto copy = [&](int column,const std::vector<float>& weights,bool device_column) {
        bool changed=false;
        for (int i=0;i<particle_count_;++i) {
            auto& previous=parameters_[static_cast<size_t>(i)*parameter_count_+column];
            if (std::memcmp(&previous,&weights[i],sizeof(float))!=0) {
                previous=weights[i];
                changed=true;
            }
        }
        if (changed) {
            if (device_column) managed_charges_dirty_=true;
            else host_parameters_dirty_=true;
        }
    };
    if (managed_charges_) copy(charge_column_,model_->latticeSum->weights(),native_charge_column_);
    if (managed_dispersion_) copy(dispersion_column_,model_->dispersion->weights(),false);
}

void CustomPairParameters::refreshStandaloneWeights(const std::map<std::string,double>& globals) {
    if (standard_ || (!managed_charges_ && !managed_dispersion_)) return;
    std::map<std::string,double> relevant;
    const auto remember = [&](const LatticeSumEvaluation& component) {
        for (const auto& name:component.globalDependencies()) relevant[name]=globals.at(name);
    };
    if (model_->latticeSum) remember(*model_->latticeSum);
    if (model_->dispersion) remember(*model_->dispersion);
    if (!weight_records_dirty_ && relevant==weight_globals_) return;
    std::optional<LatticeSumEvaluation::PreparedWeights> charges, dispersion;
    if (model_->latticeSum)
        charges=model_->latticeSum->prepareWeights(parameters_,parameter_count_,globals);
    if (model_->dispersion)
        dispersion=model_->dispersion->prepareWeights(parameters_,parameter_count_,globals);
    const auto validate = [&](const LatticeSumEvaluation::PreparedWeights& prepared) {
        for (size_t index=0;index<exclusions_.size();index+=2) {
            const float product=prepared.values[exclusions_[index]]*prepared.values[exclusions_[index+1]];
            if (!std::isfinite(product))
                throw OpenMM::OpenMMException("CustomNonbondedForce: excluded lattice weight product is not finite in FP32");
        }
    };
    if (charges) validate(*charges);
    if (dispersion) validate(*dispersion);
    // Both components and all sparse corrections are valid before either
    // authoritative weight buffer or its self/background moments are committed.
    if (charges) model_->latticeSum->commitWeights(std::move(*charges));
    if (dispersion) model_->dispersion->commitWeights(std::move(*dispersion));
    refreshManagedColumns();
    weight_globals_=std::move(relevant);
    weight_records_dirty_=false;
}

void CustomPairParameters::reorderParameters() {
    if (!atom_order_dirty_ && !host_parameters_dirty_) return;
    const auto& order=cu_.getAtomIndex();
    const bool order_changed=atom_order_dirty_ &&
        (atom_order_.size()!=static_cast<size_t>(particle_count_) ||
         !std::equal(atom_order_.begin(),atom_order_.end(),order.begin()));
    if (!order_changed && !host_parameters_dirty_) {
        atom_order_dirty_=false;
        return;
    }
    waitForUploads();
    if (order_changed && trace_updates_ && !atom_order_.empty())
        std::fprintf(stderr,"[CustomNonbondedForce] atom reorder #%u\n",++reorder_count_);
    if (order_changed)
        atom_order_.assign(order.begin(),order.begin()+particle_count_);
    reordered_.resize(parameters_.size());
    for (int i=0;i<particle_count_;++i)
        std::copy_n(parameters_.begin()+static_cast<size_t>(atom_order_[i])*parameter_count_,
                    parameter_count_,reordered_.begin()+static_cast<size_t>(i)*parameter_count_);
    const auto stream=reinterpret_cast<cudaStream_t>(cu_.getCurrentStream());
    // The expression instance receives original records at the next prepare,
    // together with that evaluation's validated globals. No prepared-field
    // layout or preprocessing dependency is visible to the plugin.
    records_pending_=true;
    if (order_changed && d_exclusions_) {
        std::vector<int> inverse(particle_count_);
        for (int i=0;i<particle_count_;++i) inverse[atom_order_[i]]=i;
        reordered_exclusions_.resize(exclusions_.size());
        for (size_t i=0;i<exclusions_.size();++i)
            reordered_exclusions_[i]=inverse[exclusions_[i]];
        upload(d_exclusions_,reordered_exclusions_.data(),
               reordered_exclusions_.size()*sizeof(int),stream);
        if (trace_updates_)
            std::fprintf(stderr,"[CustomNonbondedForce] exclusion upload #%u\n",++exclusion_upload_count_);
    }
    host_parameters_dirty_=false;
    atom_order_dirty_=false;
    // The staged full records include the latest logical charge values.
    managed_charges_dirty_=false;
}

void CustomPairParameters::setup() {
    OpenMM::ContextSelector selector(cu_);
    if (allocated_)
        throw OpenMM::OpenMMException(
            "CustomNonbondedForce: parameter buffers were already allocated; reinitialize Context");
    // Mark attempted setup too: after a partial allocation failure a fresh
    // Context is required, and the destructor releases every completed allocation.
    allocated_=true;
    const auto allocate = [](auto** pointer,size_t bytes) {
        if (bytes) cudaCheck(cudaMalloc(reinterpret_cast<void**>(pointer),bytes));
    };
    expressionCheck(cupo_expression_instantiate(program_,particle_count_,&instance_,&binding_));
    allocate(&d_exclusions_,exclusions_.size()*sizeof(int));
    atom_order_dirty_=true;
    reorderParameters();
    // DSpace creation validates topology through a host read on another stream.
    cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cu_.getCurrentStream())));
}

cupo_pair_desc CustomPairParameters::pairDescriptor() const {
    if (!instance_)
        throw OpenMM::OpenMMException("CustomNonbondedForce: allocate parameters before creating pair workspaces");
    return binding_.pair;
}

void CustomPairParameters::prepareStep(OpenMM::ContextImpl& context, cudaStream_t stream,
                                       const float* native_parameters) {
    OpenMM::ContextSelector selector(cu_);
    if (!instance_)
        throw OpenMM::OpenMMException("CustomNonbondedForce: parameter buffers are not allocated");
    if (stream!=reinterpret_cast<cudaStream_t>(cu_.getCurrentStream()))
        throw OpenMM::OpenMMException("CustomNonbondedForce: prepare parameters on OpenMM's current stream");
    const bool prepare_weights=!standard_ && (managed_charges_ || managed_dispersion_);
    std::map<std::string,double> current_globals;
    for (size_t i=0;i<global_names_.size();++i) {
        const double value=context.getParameter(global_names_[i]);
        globals_[i]=static_cast<float>(value);
        if (!std::isfinite(globals_[i]))
            throw OpenMM::OpenMMException("CustomNonbondedForce globals must be finite FP32 values");
        if (prepare_weights) current_globals[global_names_[i]]=value;
    }
    if (prepare_weights) refreshStandaloneWeights(current_globals);
    waitForUploads();
    reorderParameters();
    if (records_pending_ && managed_charges_dirty_) {
        // A full record update includes the latest canonical charges, even if
        // native lambda updates happened after setup staged the input ordering.
        for (int i=0;i<particle_count_;++i)
            reordered_[static_cast<size_t>(i)*parameter_count_+charge_column_]=
                parameters_[static_cast<size_t>(atom_order_[i])*parameter_count_+charge_column_];
    }
    cupo_expression_update update{};
    update.particles=records_pending_ && !reordered_.empty() ? reordered_.data() : nullptr;
    update.globals=globals_.empty() ? nullptr : globals_.data();
    const bool charge_update=managed_charges_dirty_ && !records_pending_ && particle_count_;
    if (charge_update) {
        if (!native_parameters)
            throw OpenMM::OpenMMException("CustomNonbondedForce: missing native managed charges");
        // Native q is already resident in the same CUDA atom order. The generic
        // column update preserves its bits without reuploading other fields.
        update.d_column=native_parameters;
        update.column_stride_bytes=3*sizeof(float);
        update.column_index=charge_column_;
    }
    expressionCheck(cupo_expression_prepare(instance_,&update,stream));
    if (trace_updates_) {
        if (records_pending_ && !reordered_.empty())
            std::fprintf(stderr,"[CustomNonbondedForce] particle upload #%u\n",++record_upload_count_);
        if (charge_update)
            std::fprintf(stderr,"[CustomNonbondedForce] managed charge update\n");
    }
    records_pending_=false;
    managed_charges_dirty_=false;
}

void CustomPairParameters::validateDefinition(const NonbondedForce& owner) const {
    const auto source=expressionDefinition(owner);
    const auto& force=*source;
    const auto& old=*definition_;
    bool changed=(owner.getPairMode()==NonbondedForce::Standard)!=standard_ ||
        (standard_ && !owner.hasCutoffExpression())!=generated_lj_ ||
        force.getNumParticles()!=particle_count_ ||
        force.getNumPerParticleParameters()!=public_parameter_count_ ||
        force.getNumGlobalParameters()!=static_cast<int>(global_names_.size()) ||
        force.getNumExclusions()!=old.getNumExclusions() ||
        force.getEnergyFunction()!=old.getEnergyFunction() ||
        force.getNonbondedMethod()!=old.getNonbondedMethod() ||
        force.getCutoffDistance()!=old.getCutoffDistance() ||
        force.getUseSwitchingFunction()!=old.getUseSwitchingFunction() ||
        (force.getUseSwitchingFunction() &&
         force.getSwitchingDistance()!=old.getSwitchingDistance()) ||
        force.getUseLongRangeCorrection()!=old.getUseLongRangeCorrection();
    for (int i=0;!changed && i<force.getNumPerParticleParameters();++i)
        changed=force.getPerParticleParameterName(i)!=particle_names_[i];
    for (size_t i=0;!changed && i<global_names_.size();++i)
        changed=force.getGlobalParameterName(i)!=global_names_[i];
    for (int i=0;!changed && i<force.getNumExclusions();++i) {
        int a,b;
        force.getExclusionParticles(i,a,b);
        changed=a!=exclusions_[2*i] || b!=exclusions_[2*i+1];
    }
    if (changed)
        throw OpenMM::OpenMMException("CustomNonbondedForce: code, layout, cutoff, switching or exclusions changed; reinitialize Context");
    std::vector<double> values;
    for (int i=0;i<particle_count_;++i) {
        force.getParticleParameters(i,values);
        if (values.size()!=static_cast<size_t>(force.getNumPerParticleParameters()))
            throw OpenMM::OpenMMException("CustomNonbondedForce: wrong number of particle parameters");
        for (double value:values)
            if (!std::isfinite(static_cast<float>(value)))
                throw OpenMM::OpenMMException("CustomNonbondedForce parameters must be finite FP32 values");
    }
}

void CustomPairParameters::copyParametersToContext(
        OpenMM::ContextImpl& context,const NonbondedForce& owner) {
    OpenMM::ContextSelector selector(cu_);
    if (generated_lj_) return; // Effective native records were supplied by the owner.
    const auto& force=owner.getCutoffExpressionDefinition();
    // Stage and validate every particle before committing any host or device
    // records. This same upload path validates initial parameters as well.
    const bool prepare_weights=!standard_ && (managed_charges_ || managed_dispersion_);
    const auto previous=prepare_weights ? parameters_ : std::vector<float>();
    const bool previous_host_dirty=host_parameters_dirty_, previous_weights_dirty=weight_records_dirty_;
    uploadParameters(force);
    if (prepare_weights) {
        try {
            std::map<std::string,double> current_globals;
            for (const auto& name:global_names_) current_globals[name]=context.getParameter(name);
            refreshStandaloneWeights(current_globals);
        } catch (...) {
            parameters_=previous;
            host_parameters_dirty_=previous_host_dirty;
            weight_records_dirty_=previous_weights_dirty;
            throw;
        }
    }
    std::vector<double> values;
    for (int i=0;i<particle_count_;++i) {
        force.getParticleParameters(i,values);
        definition_->setParticleParameters(i,values);
    }
    cu_.invalidateMolecules();
}

}  // namespace NvOpenMM
