/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file maca_common.h
 * \brief Common utilities for MACA
 */
#ifndef TVM_RUNTIME_MACA_MACA_COMMON_H_
#define TVM_RUNTIME_MACA_MACA_COMMON_H_

#include <mcr/mc_runtime_api.h>
#include <tvm/runtime/packed_func.h>

#include <string>

#include "runtime/workspace_pool.h"

namespace tvm {
namespace runtime {

#define MACA_DRIVER_CALL(x)                                                    \
  {                                                                            \
    mcError_t result = x;                                                      \
    if (result != mcSuccess && result != mcErrorDeinitialized) {               \
      LOG(FATAL) << "MACA MACA Error: " #x " failed with error: "              \
                 << mcGetErrorString(result);                                  \
    }                                                                          \
  }

#define MACA_CALL(func)                                                        \
  {                                                                            \
    mcError_t e = (func);                                                      \
    ICHECK(e == mcSuccess) << "MACA MACA: " << mcGetErrorString(e);            \
  }

/*! \brief Thread local workspace */
class MACAThreadEntry {
public:
  /*! \brief The maca stream */
  mcStream_t stream{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  MACAThreadEntry();
  // get the threadlocal workspace
  static MACAThreadEntry *ThreadLocal();
};
} // namespace runtime
} // namespace tvm
#endif // TVM_RUNTIME_MACA_MACA_COMMON_H_
