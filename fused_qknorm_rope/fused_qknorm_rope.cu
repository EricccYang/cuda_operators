#include <cuda_runtime.h>



template<typename T, int size>
struct vector_load_t;

template<>
struct vector_load<uint,4>{
    using type = uint4;
}

template<>
struct vector_load<uint,2>{
    using type = uint2;
}

template<>
struct vector_load<uint,1>{
    using type = uint;
}



__global__ void fuseQKNormRopeKernel(
    float* qkv,
    const int num_tokens,
    const int head_dim,
    const int num_q_heads,
    const int num_k_heads,
    float* rope_buffer,
    const int rotary_dim
){

    // int tid = blockDim.x
    const int warp_size = 32;
    const int block_size = 256;
    int warps_per_block = block_size / warp_size;
    
    int warp_index_in_block = threadIdx.x / warp_size;
    int warp_index= blockIdx.x * warps_per_block + warp_index_in_block;


    //load
    //warp_index -->  token-head
    //qk是怎么排列的？
    int total_heads = num_q_heads + num_k_heads + num_k_heads;
    int row_stride = total_heads * head_dim;

    int my_row =  warp_index / (num_q_heads + num_k_heads);
    float* token_head_start  =  qkv + my_row * row_stride;
    float* head_start = token_head_start + warp_index_in_block * head_dim;
    int num_elems_per_thread = head_dim / warp_size;
    //vector load

    
    vector_load<uint, num_elems_per_thread>::type vec = *reinterpret_cast<vector_load<uint, num_elems_per_thread>::type*>(head_start);
    for(int i = 0; i < num_elems_per_thread; i++){
        
    }


    //vector load
    



    //reduce compute

    

    //rope






}






// template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
//           bool interleave>
// __global__ void fusedQKNormRopeImproveKernel(
//     void* qkv_void,                  // Combined QKV tensor
//     int const num_heads_q,           // Number of query heads
//     int const num_heads_k,           // Number of key heads
//     int const num_heads_v,           // Number of value heads
//     float const eps,                 // Epsilon for RMS normalization
//     void const* q_weight_void,       // RMSNorm weights for query
//     void const* k_weight_void,       // RMSNorm weights for key
//     void const* cos_sin_cache_void,  // Pre-computed cos/sin cache
//     int64_t const* position_ids,     // Position IDs for RoPE
//     int const num_tokens,            // Number of tokens
//     int const rotary_dim             // Dimension for RoPE
// ) {
// #if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
//   if constexpr ((std::is_same_v<scalar_t_in, c10::BFloat16>) ||
//                 std::is_same_v<scalar_t_cache, c10::BFloat16>) {
//     return;
//   } else {
// #endif

//     using Converter = vllm::_typeConvert<scalar_t_in>;
//     static_assert(Converter::exists,
//                   "Input QKV data type is not supported for this CUDA "
//                   "architecture or toolkit version.");
//     using T_in = typename Converter::hip_type;
//     using T2_in = typename Converter::packed_hip_type;

//     using CacheConverter = vllm::_typeConvert<scalar_t_cache>;
//     static_assert(CacheConverter::exists,
//                   "Cache data type is not supported for this CUDA architecture "
//                   "or toolkit version.");
//     using T_cache = typename CacheConverter::hip_type;

//     T_in* qkv = reinterpret_cast<T_in*>(qkv_void);
//     T_in const* q_weight = reinterpret_cast<T_in const*>(q_weight_void);
//     T_in const* k_weight = reinterpret_cast<T_in const*>(k_weight_void);
//     T_cache const* cos_sin_cache =
//         reinterpret_cast<T_cache const*>(cos_sin_cache_void);

//     int const warpsPerBlock = blockDim.x / 32;
//     int const warpId = threadIdx.x / 32;
//     int const laneId = threadIdx.x % 32;

//     // Calculate global warp index to determine which head/token this warp
//     // processes
//     int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

//     // Total number of attention heads (Q and K)
//     int const total_qk_heads = num_heads_q + num_heads_k;

//     // Determine which token and head type (Q or K) this warp processes
//     int const tokenIdx = globalWarpIdx / total_qk_heads;
//     int const localHeadIdx = globalWarpIdx % total_qk_heads;

//     // Skip if this warp is assigned beyond the number of tokens
//     if (tokenIdx >= num_tokens) return;

//     bool const isQ = localHeadIdx < num_heads_q;
//     int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;

//     int const num_heads = num_heads_q + num_heads_k + num_heads_v;

//     static_assert(head_dim % (32 * 2) == 0,
//                   "head_dim must be divisible by 64 (each warp processes one "
//                   "head, and each thread gets even number of "
//                   "elements)");
//     constexpr int numElemsPerThread = head_dim / 32;
//     float elements[numElemsPerThread];
//     constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
//     static_assert(elemSizeBytes % 4 == 0,
//                   "numSizeBytes must be a multiple of 4");
//     constexpr int vecSize =
//         elemSizeBytes /
//         4;  // Use packed_as<uint, vecSize> to perform loading/saving.
//     using vec_T = typename tensorrt_llm::common_improve::packed_as<uint, vecSize>::type;

//     int offsetWarp;  // Offset for the warp
//     if (isQ) {
//       // Q segment: token offset + head offset within Q segment
//       offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
//     } else {
//       // K segment: token offset + entire Q segment + head offset within K
//       // segment
//       offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
//                    headIdx * head_dim;
//     }
//     int offsetThread = offsetWarp + laneId * numElemsPerThread;

//     // === Part 1: QK Norm (load, warp reduce, RMS norm, scale by weight).
//     // For per-part timing in one kernel use Nsight Compute (ncu): profile this
//     // kernel and check Source view cycle attribution for this block vs Part 2.
//     float sumOfSquares = 0.0f;
//     {
//       vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
//       constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
// #pragma unroll
//       for (int i = 0; i < num_packed_elems; i++) {
//         // Interpret the generic vector chunk as the specific packed type
//         T2_in packed_val = *(reinterpret_cast<T2_in*>(&vec) + i);
//         // Convert to float2 for computation
//         float2 vals = Converter::convert(packed_val);
//         sumOfSquares += vals.x * vals.x;
//         sumOfSquares += vals.y * vals.y;

//         elements[2 * i] = vals.x;
//         elements[2 * i + 1] = vals.y;
//       }
//     }

//     // Reduce sum across warp using the utility function
//     sumOfSquares = tensorrt_llm::common_improve::warpReduceSum(sumOfSquares);

//     // Compute RMS normalization factor
//     float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

//     // Normalize elements
// #pragma unroll
//     for (int i = 0; i < numElemsPerThread; i++) {
//       int dim = laneId * numElemsPerThread + i;
//       float weight = isQ ? Converter::convert(q_weight[dim])
//                          : Converter::convert(k_weight[dim]);
//       elements[i] *= rms_rcp * weight;
//     }

//     // === Part 2: RoPE (apply rotation to normalized Q/K, then store).
//     // NCU Source view: cycles here vs Part 1 give the ratio within this kernel.
//     float elements2[numElemsPerThread];  // Additional buffer required for RoPE.

//     int64_t pos_id = position_ids[tokenIdx];

//     // Calculate cache pointer for this position - similar to
//     // pos_encoding_kernels.cu
//     T_cache const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
//     int const embed_dim = rotary_dim / 2;
//     T_cache const* cos_ptr = cache_ptr;
//     T_cache const* sin_ptr = cache_ptr + embed_dim;
//     int const rotary_lanes = rotary_dim / numElemsPerThread;  // rotary range
//     if (laneId < rotary_lanes) {
//       if constexpr (interleave) {
//         // Perform interleaving. Use pre-computed cos/sin values.
// #pragma unroll
//         for (int i = 0; i < numElemsPerThread / 2; ++i) {
//           int const idx0 = 2 * i;
//           int const idx1 = 2 * i + 1;
//           // Global dimension index in the head
//           int const dim_idx = laneId * numElemsPerThread + idx0;

//           float const val0 = elements[idx0];
//           float const val1 = elements[idx1];

//           int const half_dim = dim_idx / 2;
//           float const cos_val =
//               CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
//           float const sin_val =
//               CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));

//           elements[idx0] = val0 * cos_val - val1 * sin_val;
//           elements[idx1] = val0 * sin_val + val1 * cos_val;
//         }
//       } else {
//         // Before data exchange with in warp, we need to sync.
//         __syncwarp();
//         int pairOffset = (rotary_dim / 2) / numElemsPerThread;
//         // Get the data from the other half of the warp. Use pre-computed
//         // cos/sin values.
// #pragma unroll
//         for (int i = 0; i < numElemsPerThread; i++) {
//           elements2[i] = __shfl_xor_sync(FINAL_MASK, elements[i], pairOffset);

//           if (laneId < pairOffset) {
//             elements2[i] = -elements2[i];
//           }
//           int dim_idx = laneId * numElemsPerThread + i;

//           dim_idx = (dim_idx * 2) % rotary_dim;
//           int half_dim = dim_idx / 2;
//           float cos_val = CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
//           float sin_val = CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));

//           elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
//         }
//         // __shfl_xor_sync does not provide memfence. Need to sync again.
//         __syncwarp();
//       }
//     }
//     // Store.
//     {
//       vec_T vec;
//       constexpr int num_packed_elems = elemSizeBytes / sizeof(T2_in);
// #pragma unroll
//       for (int i = 0; i < num_packed_elems; i++) {
//         // Convert from float2 back to the specific packed type
//         T2_in packed_val = Converter::convert(
//             make_float2(elements[2 * i], elements[2 * i + 1]));
//         // Place it into the generic vector
//         *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
//       }
//       *reinterpret_cast<vec_T*>(&qkv[offsetThread]) = vec;
//     }

// #if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
//   }
// #endif
// }


