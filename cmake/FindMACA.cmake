# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#######################################################
# Enhanced version of find maca.
#
# Usage:
#   find_maca(${USE_MACA})
#
# - When USE_MACA=ON, use auto search
# - When USE_MACA=/path/to/maca-sdk-path, use the sdk
#
# Provide variables:
#
# - MACA_FOUND
# - MACA_INCLUDE_DIRS
# - MACA_MACAMCC_LIBRARY

macro(find_maca use_maca)
  set(__use_maca ${use_maca})
  if(IS_DIRECTORY ${__use_maca})
    set(__maca_sdk ${__use_maca})
    message(STATUS "Custom MACA SDK PATH=" ${__use_maca})
  elseif(IS_DIRECTORY $ENV{MACA_PATH})
    set(__maca_sdk $ENV{MACA_PATH})
  elseif(IS_DIRECTORY /opt/maca)
    set(__maca_sdk /opt/maca)
  else()
    set(__maca_sdk "")
  endif()

  if(__maca_sdk)
    set(MACA_INCLUDE_DIRS ${__maca_sdk}/include)
    find_library(MACA_MACAMCC_LIBRARY mcruntime ${__maca_sdk}/lib)
    find_library(MACA_HCA_LIBRARY mxc-runtime64 ${__maca_sdk}/lib)

    if(MACA_MACAMCC_LIBRARY)
      set(MACA_FOUND TRUE)
    endif()
  endif(__maca_sdk)
  if(MACA_FOUND)
    message(STATUS "Found MACA_INCLUDE_DIRS=" ${MACA_INCLUDE_DIRS})
    message(STATUS "Found MACA_MACAMCC_LIBRARY=" ${MACA_MACAMCC_LIBRARY})
  endif(MACA_FOUND)
endmacro(find_maca)
