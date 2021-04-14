/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "SiLUPlugin.h"
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::PluginType;
using nvinfer1::plugin::LReluPluginCreator;
using nvinfer1::plugin::SiLU;

static const char* LRELU_PLUGIN_VERSION{"1"};
static const char* LRELU_PLUGIN_NAME{"SiLU"};
PluginFieldCollection SiLUPluginCreator::mFC{};
std::vector<PluginField> SiLUPluginCreator::mPluginAttributes;

SiLU::SiLU() {}

SiLU::SiLU(nvinfer1::DataType iType, int iC, int iH, int iW)
    : iType(iType)
    , iC(iC)
    , iH(iH)
    , iW(iW)
{
}

SiLU::SiLU(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    iC = read<int>(d);
    iH = read<int>(d);
    iW = read<int>(d);
    ASSERT(d == a + length);
}

int SiLU::getNbOutputs() const
{
    return 1;
}

Dims SiLU::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return inputs[0];
}

size_t SiLU::getSerializationSize() const
{
    return 0;
}

void SiLU::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mNegSlope);
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

void SiLU::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, onst bool* inputIsBroadcast, const bool* outputIsBroadcast,
    nvinfer1::PluginFormat format, int maxBatchSize)
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    iC = inputDims->d[0];
    iH = inputDims->d[1];
    iW = inputDims->d[2];

    iType = inputTypes[0];
}

bool SiLU::supportsFormat(DataType type, PluginFormat format) const
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
}

int SiLU::initialize()
{
    return STATUS_SUCCESS;
}

void SiLU::terminate() {}

size_t SiLU::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

const char* SiLU::getPluginType() const
{
    return LRELU_PLUGIN_NAME;
}

const char* SiLU::getPluginVersion() const
{
    return LRELU_PLUGIN_VERSION;
}

void SiLU::destroy()
{
    delete this;
}

IPluginV2* SiLU::clone() const
{
    IPluginV2* plugin = new SiLU();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

bool SiLUPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Plugin creator
SiLUPluginCreator::SiLUPluginCreator() {}

const char* SiLUPluginCreator::getPluginName() const
{
    return SILU_PLUGIN_NAME;
}

const char* SiluPluginCreator::getPluginVersion() const
{
    return SILU_PLUGIN_VERSION;
}

const PluginFieldCollection* SiLUPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* SiLUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    SiLUPlugin* plugin = new SiLUPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2* SiLUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call LReluPlugin::destroy()
    SiLUPlugin* plugin = new SiLUPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
// LeakReLU }}}
