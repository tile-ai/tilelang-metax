// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights
// reserved.

#include "../runtime/maca_module.h"
#include "codegen_maca.h"
#include "op/builtin.h"
#include "runtime/pack_args.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace codegen {

using namespace ffi;

static std::string GetDeviceGlobalSymbol(const GlobalVar &gvar,
                                         const tirx::PrimFunc &f) {
  if (auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
    return static_cast<std::string>(global_symbol.value());
  }
  return gvar->name_hint;
}

static void ValidateUniqueDeviceGlobalSymbols(const IRModule &mod) {
  std::unordered_map<std::string, std::string> symbol_to_gvar;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tirx::PrimFunc>(kv.second);
    std::string global_symbol = GetDeviceGlobalSymbol(gvar, f);

    auto [it, inserted] =
        symbol_to_gvar.emplace(global_symbol, gvar->name_hint);
    ICHECK(inserted)
        << "Duplicate MACA kernel global_symbol `" << global_symbol
        << "` found on PrimFuncs `" << it->second << "` and `"
        << gvar->name_hint
        << "`. T.MACASourceCodeKernel emits raw MACA source without "
           "renaming, so MACA entry names must be unique within the compiled "
           "module.";
  }
}

static Map<String, runtime::FunctionInfo> ExtractFuncInfo(const IRModule &mod) {
  Map<String, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tirx::PrimFunc>(kv.second);

    Array<DLDataType> arg_types;
    Array<String> launch_param_tags;

    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (f->params[i]->dtype.is_handle()) {
        auto ptr = f->params[i]->type_annotation.as<PointerTypeNode>();
        if (ptr && ptr->storage_scope == "grid_constant") {
          arg_types.push_back(DataType(runtime::kDLGridConstant, 64, 1));
          continue;
        }
      }
      DataType dtype = f->params[i].dtype();
      // Device runtime cannot directly take bool arguments, map to int32.
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      arg_types.push_back(dtype);
    }
    if (f->HasNonzeroAttr(tl::attr::kHasGridSync)) {
      launch_param_tags.push_back(
          runtime::launch_param::kUseProgramaticDependentLaunch);
    }
    if (f->HasNonzeroAttr("use_cooperative_groups")) {
      launch_param_tags.push_back(runtime::launch_param::kUseCooperativeLaunch);
    }
    if (f->GetAttr<Array<Integer>>("cluster_dims").defined()) {
      launch_param_tags.push_back(runtime::launch_param::kClusterDimX);
      launch_param_tags.push_back(runtime::launch_param::kClusterDimY);
      launch_param_tags.push_back(runtime::launch_param::kClusterDimZ);
    }
    if (auto opt = f->GetAttr<Array<String>>(tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        if (tag != runtime::launch_param::kClusterDimX &&
            tag != runtime::launch_param::kClusterDimY &&
            tag != runtime::launch_param::kClusterDimZ) {
          launch_param_tags.push_back(tag);
        }
      }
    }
    std::string sym = GetDeviceGlobalSymbol(Downcast<GlobalVar>(kv.first), f);
    fmap.Set(String(sym), runtime::FunctionInfo(String(sym), arg_types,
                                                launch_param_tags, {}));
  }
  return fmap;
}

Module BuildTileLangMACA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangMACA cg;
  cg.Init(output_ssa);
  cg.SetEmitLineDirectives(
      tvm::transform::PassContext::Current()
          ->GetConfig<Bool>(tl::kEmitLineDirectives, Bool(false))
          .value());

  ValidateUniqueDeviceGlobalSymbols(mod);
  if (const auto f = Function::GetGlobal("tilelang_callback_maca_validate")) {
    (*f)(mod);
  }

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangMACA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_maca_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  std::string fmt = "mcir";
  std::string mcir;
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_maca_compile")) {
    // Fetch current pass context config and pass into the compile callback
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    mcir = (*f)(code, target, pass_ctx->config).cast<std::string>();
    if (mcir[0] != '/')
      fmt = "mcbin";
  } else {
    ICHECK(false) << "tilelang_callback_maca_compile is not set";
  }
  return runtime::MACAModuleCreate(mcir, fmt, ExtractFuncInfo(mod), code);
}

ffi::Module BuildTileLangMACAWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangMACA cg;
  cg.Init(output_ssa);
  cg.SetEmitLineDirectives(
      tvm::transform::PassContext::Current()
          ->GetConfig<Bool>(tl::kEmitLineDirectives, Bool(false))
          .value());

  ValidateUniqueDeviceGlobalSymbols(mod);
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_maca_validate")) {
    (*f)(mod);
  }

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangMACA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_maca_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  return runtime::MACAModuleCreate("mcir", "mcir", ExtractFuncInfo(mod), code);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_maca", BuildTileLangMACA)
      .def("target.build.tilelang_maca_without_compile",
           BuildTileLangMACAWithoutCompile);
}

} // namespace codegen
} // namespace tvm
