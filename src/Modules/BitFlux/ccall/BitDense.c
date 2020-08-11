#include <stdint.h>
#include <math.h>
#include <stdio.h>

void setBit(uint64_t* in, int i, int value)
{
    const uint64_t one = 1;
    int chunk = floor(i / 64.0);
    int jMod = i % 64;

    in[chunk] &= ~(one << jMod);
    in[chunk] |= (((uint64_t) value) << jMod);
}

int getBit(uint64_t* in, int i)
{
    const uint64_t one = 1;
    int chunk = floor(i / 64.0);
    int jMod = i % 64;
    
    return (in[chunk] & (one << jMod)) >> jMod;
}

void binaryBitOutDotProduct(uint64_t* in, uint64_t* weights, uint64_t* bias, uint64_t* out, int neuronNumber, int wNumber)
{
    int wNumberChunks = ceil(wNumber / 64.0);
    
    if(wNumberChunks > 1)
    {
        int tempOut[neuronNumber];
        int iLim = wNumberChunks - 1;

        uint64_t firstIn = in[0];
        for(int i = 0; i < neuronNumber; i++)
            tempOut[i] = __builtin_popcountl(firstIn ^ weights[i]);
        weights += neuronNumber;

        for(int i = 1; i < iLim; i++)
        {
            for(int j = 0; j < neuronNumber; j++)
                tempOut[j] += __builtin_popcountl(in[i] ^ weights[j]);
            weights += neuronNumber;
        }

        uint64_t lastIn = in[iLim];
        for(int i = 0; i < neuronNumber; i++)
        {
            int b = getBit(bias, i);
            tempOut[i] += __builtin_popcountl(lastIn ^ weights[i]);
            setBit(out, i, ((tempOut[i] + b) > ((wNumber + !b) - tempOut[i])));
        }
    }
    else
    {
        for(int i = 0; i < neuronNumber; i++)
        {
            int b = getBit(bias, i);
            int temp = __builtin_popcountl(in[0] ^ weights[i]);
            setBit(out, i, ((temp + b) > ((wNumber + !b) - temp)));
        }
    }
}

void binaryFloatOutDotProduct(uint64_t* in, uint64_t* weights, uint64_t* bias, float* out, int neuronNumber, int wNumber)
{
    int wNumberChunks = ceil(wNumber / 64.0);
    
    if(wNumberChunks > 1)
    {
        int tempOut[neuronNumber];
        int iLim = wNumberChunks - 1;

        uint64_t firstIn = in[0];
        for(int i = 0; i < neuronNumber; i++)
            tempOut[i] = __builtin_popcountl(firstIn ^ weights[i]);
        weights += neuronNumber;

        for(int i = 1; i < iLim; i++)
        {
            for(int j = 0; j < neuronNumber; j++)
                tempOut[j] += __builtin_popcountl(in[i] ^ weights[j]);
            weights += neuronNumber;
        }

        uint64_t lastIn = in[iLim];
        for(int i = 0; i < neuronNumber; i++)
        {
            int b = getBit(bias, i);
            tempOut[i] += __builtin_popcountl(lastIn ^ weights[i]);
            out[i] = (1 - (tempOut[i] + b) / ((double) wNumber)) * 2 - 1;
        }
    }
    else
    {
        for(int i = 0; i < neuronNumber; i++)
        {
            int b = getBit(bias, i);
            int temp = __builtin_popcountl(in[0] ^ weights[i]);
            out[i] = (1 - (temp + b) / ((double) wNumber)) * 2 - 1;
        }
    }
}