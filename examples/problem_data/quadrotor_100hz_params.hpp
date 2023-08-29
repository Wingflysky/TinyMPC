#pragma once

#include <tinympc/types.hpp>

tinytype rho_value = 5.0;

tinytype Adyn_data[NSTATES*NSTATES] = {
  1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0009810,	0.0000000,	0.0100000,	0.0000000,	0.0000000,	0.0000000,	0.0000016,	0.0000000,	
  0.0000000,	1.0000000,	0.0000000,	-0.0009810,	0.0000000,	0.0000000,	0.0000000,	0.0100000,	0.0000000,	-0.0000016,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0100000,	0.0000000,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0050000,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0050000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0050000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.1962000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0004905,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	-0.1962000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	-0.0004905,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000	
};

tinytype Bdyn_data[NSTATES*NINPUTS] = {
  -0.0000011,	0.0000012,	0.0000011,	-0.0000012,	
  0.0000011,	0.0000012,	-0.0000011,	-0.0000012,	
  0.0002102,	0.0002102,	0.0002102,	0.0002102,	
  -0.0068839,	-0.0075809,	0.0068916,	0.0075732,	
  -0.0069177,	0.0076070,	0.0069392,	-0.0076285,	
  0.0004937,	-0.0001806,	-0.0006961,	0.0003830,	
  -0.0004524,	0.0004975,	0.0004538,	-0.0004989,	
  0.0004502,	0.0004958,	-0.0004507,	-0.0004953,	
  0.0420429,	0.0420429,	0.0420429,	0.0420429,	
  -2.7535461,	-3.0323404,	2.7566264,	3.0292601,	
  -2.7670702,	3.0427842,	2.7756950,	-3.0514090,	
  0.1974771,	-0.0722364,	-0.2784376,	0.1531969	
};

tinytype Kinf_data[NINPUTS*NSTATES] = {
  -0.3906311,	0.3517458,	1.6138691,	-1.4145659,	-1.6408468,	-3.1448466,	-0.2790642,	0.2472196,	0.6443451,	-0.1065374,	-0.1300801,	-0.6454452,	
  0.3638961,	0.2700654,	1.6138691,	-0.9970911,	1.5362881,	3.1624407,	0.2604559,	0.1840642,	0.6443451,	-0.0668165,	0.1225141,	0.6476988,	
  0.2533040,	-0.2968801,	1.6138691,	1.1019185,	0.8586152,	-3.2017226,	0.1677149,	-0.2027246,	0.6443451,	0.0744003,	0.0496951,	-0.6526659,	
  -0.2265691,	-0.3249312,	1.6138691,	1.3097384,	-0.7540564,	3.1841284,	-0.1491066,	-0.2285593,	0.6443451,	0.0989536,	-0.0421290,	0.6504123	
};

tinytype Pinf_data[NSTATES*NSTATES] = {
  7285.7896800,	-1.7192050,	0.0000000,	7.0399783,	5665.2347850,	102.4288880,	2041.4498269,	-1.2358674,	0.0000000,	0.4852510,	15.7425023,	15.7433820,	
  -1.7192050,	7282.9516748,	-0.0000000,	-5653.3797772,	-7.0403703,	-40.9487287,	-1.2358998,	2039.3863263,	-0.0000000,	-14.9264995,	-0.4852885,	-6.2941346,	
  0.0000000,	-0.0000000,	4192.1760488,	-0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	365.9124543,	-0.0000000,	0.0000000,	0.0000000,	
  7.0399783,	-5653.3797772,	0.0000000,	21144.6563411,	33.1412099,	211.1309209,	5.3386519,	-3862.8800132,	0.0000000,	56.8424049,	2.5850230,	35.7262836,	
  5665.2347850,	-7.0403703,	0.0000000,	33.1412099,	21203.0012197,	528.0361828,	3872.0644251,	-5.3387987,	0.0000000,	2.5849839,	61.4824326,	89.3448045,	
  102.4288880,	-40.9487287,	-0.0000000,	211.1309209,	528.0361828,	16557.6187885,	80.4455051,	-32.1627169,	-0.0000000,	17.9596832,	44.9133550,	751.2088882,	
  2041.4498269,	-1.2358998,	0.0000000,	5.3386519,	3872.0644251,	80.4455051,	1122.2793231,	-0.9078133,	0.0000000,	0.3821186,	10.9348345,	12.7462947,	
  -1.2358674,	2039.3863263,	-0.0000000,	-3862.8800132,	-5.3387987,	-32.1627169,	-0.9078133,	1120.7398856,	-0.0000000,	-10.2732855,	-0.3821359,	-5.0963418,	
  0.0000000,	-0.0000000,	365.9124543,	-0.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0000000,	148.7627641,	-0.0000000,	0.0000000,	0.0000000,	
  0.4852510,	-14.9264995,	-0.0000000,	56.8424049,	2.5849839,	17.9596832,	0.3821186,	-10.2732855,	-0.0000000,	7.5408433,	0.2528901,	3.6573630,	
  15.7425023,	-0.4852885,	0.0000000,	2.5850230,	61.4824326,	44.9133550,	10.9348345,	-0.3821359,	0.0000000,	0.2528901,	8.0152054,	9.1453135,	
  15.7433820,	-6.2941346,	-0.0000000,	35.7262836,	89.3448045,	751.2088882,	12.7462947,	-5.0963418,	-0.0000000,	3.6573630,	9.1453135,	156.2694404	
};

tinytype Quu_inv_data[NINPUTS*NINPUTS] = {
  0.0518570,	-0.0002002,	0.0477301,	-0.0001651,	
  -0.0002002,	0.0514551,	-0.0000809,	0.0480478,	
  0.0477301,	-0.0000809,	0.0515671,	0.0000055,	
  -0.0001651,	0.0480478,	0.0000055,	0.0513336	
};

tinytype AmBKt_data[NSTATES*NSTATES] = {
  0.9999985,	-0.0000000,	0.0000000,	0.0000398,	-0.0089565,	0.0005217,	-0.0005858,	-0.0000026,	0.0000000,	0.0159084,	-3.5826108,	0.2086663,	
  -0.0000000,	0.9999985,	0.0000000,	0.0089754,	-0.0000397,	-0.0002071,	-0.0000026,	-0.0005870,	0.0000000,	3.5901672,	-0.0158948,	-0.0828373,	
  0.0000000,	0.0000000,	0.9986430,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.2714067,	0.0000000,	0.0000000,	0.0000000,	
  0.0000000,	-0.0009753,	-0.0000000,	0.9651907,	0.0001442,	0.0007837,	0.0000094,	-0.1939235,	-0.0000000,	-13.9237080,	0.0576876,	0.3134858,	
  0.0009753,	-0.0000000,	0.0000000,	0.0001444,	0.9652522,	0.0019740,	0.1939275,	-0.0000094,	0.0000000,	0.0577526,	-13.8991197,	0.7895955,	
  0.0000001,	-0.0000000,	0.0000000,	0.0002763,	0.0006961,	0.9986755,	0.0000455,	-0.0000181,	0.0000000,	0.1105165,	0.2784478,	-0.5298000,	
  0.0099990,	-0.0000000,	0.0000000,	0.0000268,	-0.0062130,	0.0003587,	0.9995937,	-0.0000018,	0.0000000,	0.0107303,	-2.4852121,	0.1434640,	
  -0.0000000,	0.0099990,	0.0000000,	0.0062252,	-0.0000268,	-0.0001424,	-0.0000018,	0.9995929,	0.0000000,	2.4900775,	-0.0107198,	-0.0569557,	
  -0.0000000,	0.0000000,	0.0094582,	0.0000000,	0.0000000,	0.0000000,	-0.0000000,	0.0000000,	0.8916396,	0.0000000,	-0.0000000,	0.0000000,	
  0.0000000,	-0.0000012,	-0.0000000,	0.0024980,	0.0000099,	0.0000544,	0.0000006,	-0.0003269,	-0.0000000,	-0.0008160,	0.0039471,	0.0217686,	
  0.0000012,	-0.0000000,	0.0000000,	0.0000099,	0.0025020,	0.0001371,	0.0003271,	-0.0000006,	0.0000000,	0.0039520,	0.0007839,	0.0548288,	
  0.0000000,	-0.0000000,	0.0000000,	0.0000392,	0.0000987,	0.0047322,	0.0000065,	-0.0000026,	0.0000000,	0.0156682,	0.0394755,	0.8928802	
};

tinytype coeff_d2p_data[NSTATES*NINPUTS] = {
  0.0000000,	-0.0000000,	-0.0000000,	0.0000000,	
  -0.0000000,	-0.0000000,	0.0000000,	0.0000000,	
  0.0000000,	-0.0000000,	0.0000000,	0.0000000,	
  -0.0000000,	0.0000000,	0.0000000,	-0.0000000,	
  0.0000000,	-0.0000000,	0.0000000,	-0.0000000,	
  -0.0000000,	-0.0000000,	-0.0000000,	0.0000000,	
  0.0000000,	-0.0000000,	-0.0000000,	0.0000000,	
  -0.0000000,	-0.0000000,	0.0000000,	0.0000000,	
  0.0000000,	-0.0000000,	0.0000000,	0.0000000,	
  -0.0000000,	-0.0000000,	0.0000000,	0.0000000,	
  -0.0000000,	-0.0000000,	0.0000000,	0.0000000,	
  -0.0000000,	-0.0000000,	-0.0000000,	0.0000000	
};

tinytype Q_data[NSTATES]= {100.0000000,	100.0000000,	100.0000000,	4.0000000,	4.0000000,	400.0000000,	4.0000000,	4.0000000,	4.0000000,	2.0408163,	2.0408163,	4.0000000};

tinytype Qf_data[NSTATES]= {100.0000000,	100.0000000,	100.0000000,	4.0000000,	4.0000000,	400.0000000,	4.0000000,	4.0000000,	4.0000000,	2.0408163,	2.0408163,	4.0000000};

tinytype R_data[NINPUTS]= {4.0000000,	4.0000000,	4.0000000,	4.0000000};

