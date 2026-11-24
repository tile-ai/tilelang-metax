#include <algorithm>
#include <cctype>
#include <charconv>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

namespace refl = ffi::reflection;

std::string ExtractStringWithPrefix(const std::string &str,
                                    const std::string &prefix) {
  if (str.find(prefix) != 0)
    return "";
  std::size_t pos = prefix.length();
  while (pos < str.length() &&
         (std::isdigit(str[pos]) || std::isalpha(str[pos]))) {
    ++pos;
  }
  return str.substr(prefix.length(), pos - prefix.length());
}

void CheckOrSetAttr(ffi::Map<ffi::String, ffi::Any> *attrs,
                    const ffi::String &name, const ffi::String &value) {
  auto iter = attrs->find(name);
  if (iter == attrs->end()) {
    attrs->Set(name, value);
  } else {
    auto str = (*iter).second.try_cast<ffi::String>();
    TVM_FFI_CHECK(str && str.value() == value, ValueError)
        << "Expects \"" << name << "\" to be \"" << value
        << "\", but gets: " << (*iter).second;
  }
}

/*!
 * \brief Update the attributes in the LLVM MACA target.
 * \param target The Target to update
 * \return The updated attributes
 */
ffi::Map<ffi::String, ffi::Any>
UpdateMACAAttrs(ffi::Map<ffi::String, ffi::Any> target) {
  CheckOrSetAttr(&target, "mtriple", "mxc-metax-macahca");
  // Update -mcpu=xcore1000
  std::string arch = "xcore1000";
  if (target.count("mcpu")) {
    ffi::String mcpu = Downcast<ffi::String>(target.at("mcpu"));
    arch = ExtractStringWithPrefix(mcpu, "xcore");
    TVM_FFI_CHECK(!arch.empty(), ValueError)
        << "MACA target gets an invalid XCORE version: -mcpu=" << mcpu;
  } else {
    if (auto f_get_maca_arch =
            tvm::ffi::Function::GetGlobal("tvm_callback_maca_get_arch")) {
      arch = (*f_get_maca_arch)().cast<std::string>();
    }
    target.Set("mcpu", ffi::String(arch));
  }

  if (!target.count("xcore_num")) {
    int64_t xcore_num = 1000;
    auto first_digit =
        std::find_if(arch.begin(), arch.end(),
                     [](unsigned char c) { return std::isdigit(c); });
    TVM_FFI_CHECK(first_digit != arch.end(), ValueError)
        << "MACA target gets an invalid XCORE version: -mcpu=" << arch;
    auto [ptr, ec] =
        std::from_chars(&*first_digit, arch.data() + arch.size(), xcore_num);
    TVM_FFI_CHECK(ec == std::errc{}, ValueError)
        << "MACA target gets an invalid XCORE version: -mcpu=" << arch;
    target.Set("xcore_num", xcore_num);
  }

  return target;
}

TVM_REGISTER_TARGET_KIND("maca", kDLMACA)
    .add_attr_option<ffi::String>("mcpu")
    .add_attr_option<int64_t>("xcore_num")
    .add_attr_option<ffi::String>("mtriple")
    .add_attr_option<ffi::Array<ffi::String>>("mattr")
    .add_attr_option<int64_t>("max_num_threads", refl::DefaultValue(1024))
    .add_attr_option<int64_t>("max_threads_per_block", refl::DefaultValue(1024))
    .add_attr_option<int64_t>("max_shared_memory_per_block",
                              refl::DefaultValue(65536))
    .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(64))
    .add_attr_option<int64_t>("max_local_memory_per_block",
                              refl::DefaultValue(4095))
    .set_default_keys({"maca", "gpu"})
    .set_target_canonicalizer(UpdateMACAAttrs);

} // namespace tvm
