#include "kgraph.h"
#include "kgraph-data.h"

#ifdef __AVX__
#include <immintrin.h>
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_loadu_ps(addr1);\
    tmp2 = _mm256_loadu_ps(addr2);\
    tmp1 = _mm256_sub_ps(tmp1, tmp2); \
    tmp1 = _mm256_mul_ps(tmp1, tmp1); \
    dest = _mm256_add_ps(dest, tmp1); 
namespace kgraph {
float float_l2sqr_avx (float const *t1, float const *t2, unsigned dim) {
    __m256 sum;
    __m256 l0, l1, l2, l3;
    __m256 r0, r1, r2, r3;
    unsigned D = (dim + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
    unsigned DR = D % 32;
    unsigned DD = D - DR;
    const float *l = t1;
    const float *r = t2;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    alignas(32) float unpack[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float ret = 0.0;
    sum = _mm256_load_ps(unpack);
    switch (DR) {
        case 24:
            AVX_L2SQR(e_l+16, e_r+16, sum, l2, r2);
        case 16:
            AVX_L2SQR(e_l+8, e_r+8, sum, l1, r1);
        case 8:
            AVX_L2SQR(e_l, e_r, sum, l0, r0);
    }
    for (unsigned i = 0; i < DD; i += 32, l += 32, r += 32) {
        AVX_L2SQR(l, r, sum, l0, r0);
        AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
        AVX_L2SQR(l + 16, r + 16, sum, l2, r2);
        AVX_L2SQR(l + 24, r + 24, sum, l3, r3);
    }
    _mm256_storeu_ps(unpack, sum);
    ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    return ret;//sqrt(ret);
}
}
#endif
#ifdef __SSE2__
#include <xmmintrin.h>
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_load_ps(addr1);\
    tmp2 = _mm_load_ps(addr2);\
    tmp1 = _mm_sub_ps(tmp1, tmp2); \
    tmp1 = _mm_mul_ps(tmp1, tmp1); \
    dest = _mm_add_ps(dest, tmp1); 
namespace kgraph {
float float_l2sqr_sse2 (float const *t1, float const *t2, unsigned dim) {
	/*	
	//cos dostance
	float dis=0;
	float x_ = 0;
	float y_ = 0;   
	for(unsigned i=0;i<dim;i++)
	{
		dis += t1[i]*t2[i];
		x_ += t1[i]*t1[i];
		y_ += t2[i]*t2[i];
	}
	dis = dis / sqrt(x_*y_);
	cout<<dis<<endl;
	return dis;
	*/

    //l2 distance
	///*	
	float dis=0;   
    for(unsigned i=0;i<dim;i++)
    {
        //dis += pow((t1[i]-t2[i]),2);
        float tmp = t1[i]-t2[i];
        dis += tmp * tmp;
    }
    return dis;
	//*/
	
	// about the hardware speed
	/* 
    #ifdef DISTANCE_SP 
    __m128 sum;
    __m128 l0, l1, l2, l3;
    __m128 r0, r1, r2, r3;
    unsigned D = (dim + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = t1;
    const float *r = t2;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[4] __attribute__ ((aligned (16))) = {0, 0, 0, 0};
    float ret = 0.0;
    sum = _mm_load_ps(unpack);
    switch (DR) {
        case 12:
            SSE_L2SQR(e_l+8, e_r+8, sum, l2, r2);
        case 8:
            SSE_L2SQR(e_l+4, e_r+4, sum, l1, r1);
        case 4:
            SSE_L2SQR(e_l, e_r, sum, l0, r0);
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
        SSE_L2SQR(l, r, sum, l0, r0);
        SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
        SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
        SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
    return ret;//sqrt(ret);	
    #else 
    float dis=0;   
    for(unsigned i=0;i<dim;i++)
    {
        //dis += pow((t1[i]-t2[i]),2);
        float tmp = t1[i]-t2[i];
        dis += tmp * tmp;
    }
    return dis;
    #endif 
*/


}
}
/*
template <typename T>
void print_128 (__m128i v) {
    static unsigned constexpr L = 16 / sizeof(T);
    T unpack[L] __attribute__ ((aligned (16)));
    _mm_store_si128((__m128i *)unpack, v);
    cout << '(' << int(unpack[0]);
    for (unsigned i = 1; i < L; ++i) {
        cout << ',' << int(unpack[i]);
    }
    cout << ')';
}
*/

#define SSE_L2SQR_BYTE(addr1, addr2, sum, z) \
    do { \
        const __m128i o = _mm_load_si128((__m128i const *)(addr1));\
        const __m128i p = _mm_load_si128((__m128i const *)(addr2));\
        __m128i o1 = _mm_unpackhi_epi8(o,z); \
        __m128i p1 = _mm_unpackhi_epi8(p,z); \
        __m128i d = _mm_sub_epi16(o1, p1); \
        sum = _mm_add_epi32(sum, _mm_madd_epi16(d, d)); \
        o1 = _mm_unpacklo_epi8(o,z); \
        p1 = _mm_unpacklo_epi8(p,z); \
        d = _mm_sub_epi16(o1, p1); \
        sum = _mm_add_epi32(sum, _mm_madd_epi16(d, d)); \
    } while (false)
namespace kgraph {
float uint8_l2sqr_sse2 (uint8_t const *t1, uint8_t const *t2, unsigned dim) {
	cout<<"3\n";    
	unsigned D = (dim + 0xFU) & ~0xFU;   // actual dimension used in calculation, 0-padded
    unsigned DR = D % 64;           // process 32 dims per iteration
    unsigned DD = D - DR;
    const uint8_t *l = t1;
    const uint8_t *r = t2;
    const uint8_t *e_l = l + DD;
    const uint8_t *e_r = r + DD;
    alignas(16) int32_t unpack[4] = {0, 0, 0, 0};
    __m128i sum = _mm_load_si128((__m128i *)unpack);
    const __m128i z = sum;
    switch (DR) {
        case 48:
            SSE_L2SQR_BYTE(e_l+32, e_r+32, sum, z);
        case 32:
            SSE_L2SQR_BYTE(e_l+16, e_r+16, sum, z);
        case 16:
            SSE_L2SQR_BYTE(e_l, e_r, sum, z);
    }
    for (unsigned i = 0; i < DD; i += 64, l += 64, r += 64) {
        SSE_L2SQR_BYTE(l, r, sum, z);
        SSE_L2SQR_BYTE(l + 16, r + 16, sum, z);
        SSE_L2SQR_BYTE(l + 32, r + 32, sum, z);
        SSE_L2SQR_BYTE(l + 48, r + 48, sum, z);
    }
    _mm_store_si128((__m128i *)unpack, sum);
    int32_t ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
    return float(ret);//sqrt(ret);
}
}
#endif
