/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#ifndef SD_PLATFORM_BOILERPLATE_H
#define SD_PLATFORM_BOILERPLATE_H

#define PLATFORM_IMPL(NAME)         class ND4J_EXPORT PLATFORM_##NAME : public PlatformHelper {\
                                    public: \
                                        PLATFORM_##NAME() :  PlatformHelper(#NAME) { } \
                                        bool isUsable(graph::Context &context) override; \
                                        Nd4jStatus invokeHelper(graph::Context &context) override; \
                                    }; \
                                    struct __registratorHelper_##NAME { \
                                        __registratorHelper_##NAME() { \
                                            nd4j_printf("Registrator invoked\n", ""); \
                                            auto helper = new PLATFORM_##NAME(); \
                                            OpRegistrator::getInstance()->registerHelper(helper); \
                                        } \
                                    }; \
                                    static __registratorHelper_##NAME platformHelper##NAME; \
                                    Nd4jStatus PLATFORM_##NAME::invokeHelper(nd4j::graph::Context &context)


#define PLATFORM_CHECK(NAME)        bool PLATFORM_##NAME::isUsable(graph::Context &context)


#endif //SD_PLATFORM_BOILERPLATE_H
