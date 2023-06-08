/*
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023
*/


static float LUT_EXP[7][256] = {
       {
1.603811e-28, 2.644237e-28, 4.359610e-28, 7.187782e-28, 1.185065e-27, 1.953842e-27, 3.221340e-27, 5.311092e-27, 8.756511e-27, 1.443705e-26, 2.380266e-26, 3.924396e-26, 6.470235e-26, 1.066761e-25, 1.758792e-25, 2.899758e-25, 4.780893e-25, 7.882360e-25, 1.299581e-24, 2.142648e-24, 3.532629e-24, 5.824320e-24, 9.602680e-24, 1.583214e-23, 2.610279e-23, 4.303622e-23, 7.095474e-23, 1.169846e-22, 1.928750e-22, 3.179971e-22, 5.242886e-22, 8.644057e-22, 1.425164e-21, 2.349698e-21, 3.873998e-21, 6.387143e-21, 1.053062e-20, 1.736205e-20, 2.862519e-20, 4.719495e-20, 7.781132e-20, 1.282892e-19, 2.115131e-19, 3.487262e-19, 5.749523e-19, 9.479360e-19, 1.562882e-18, 2.576757e-18, 4.248354e-18, 7.004352e-18, 1.154822e-17, 1.903980e-17, 3.139133e-17, 5.175555e-17, 8.533048e-17, 1.406862e-16, 2.319523e-16, 3.824247e-16, 6.305117e-16, 1.039538e-15, 1.713908e-15, 2.825757e-15, 4.658886e-15, 7.681205e-15, 1.266417e-14, 2.087968e-14, 3.442477e-14, 5.675685e-14, 9.357623e-14, 1.542811e-13, 2.543666e-13, 4.193796e-13, 6.914400e-13, 1.139992e-12, 1.879529e-12, 3.098819e-12, 5.109089e-12, 8.423464e-12, 1.388794e-11, 2.289735e-11, 3.775135e-11, 6.224145e-11, 1.026188e-10, 1.691898e-10, 2.789468e-10, 4.599056e-10, 7.582560e-10, 1.250153e-09, 2.061154e-09, 3.398268e-09, 5.602796e-09, 9.237450e-09, 1.522998e-08, 2.510999e-08, 4.139938e-08, 6.825604e-08, 1.125352e-07, 1.855391e-07, 3.059023e-07, 5.043477e-07, 8.315287e-07, 1.370959e-06, 2.260329e-06, 3.726653e-06, 6.144212e-06, 1.013009e-05, 1.670170e-05, 2.753645e-05, 4.539993e-05, 7.485183e-05, 1.234098e-04, 2.034684e-04, 3.354626e-04, 5.530844e-04, 9.118820e-04, 1.503439e-03, 2.478752e-03, 4.086772e-03, 6.737947e-03, 1.110900e-02, 1.831564e-02, 3.019738e-02, 4.978707e-02, 8.208500e-02, 1.353353e-01, 2.231302e-01, 3.678795e-01, 6.065307e-01, 1.000000e+00, 1.648721e+00, 2.718282e+00, 4.481689e+00, 7.389056e+00, 1.218249e+01, 2.008554e+01, 3.311545e+01, 5.459815e+01, 9.001714e+01, 1.484132e+02, 2.446919e+02, 4.034288e+02, 6.651416e+02, 1.096633e+03, 1.808042e+03, 2.980958e+03, 4.914769e+03, 8.103084e+03, 1.335973e+04, 2.202646e+04, 3.631550e+04, 5.987414e+04, 9.871577e+04, 1.627548e+05, 2.683373e+05, 4.424134e+05, 7.294164e+05, 1.202604e+06, 1.982759e+06, 3.269017e+06, 5.389698e+06, 8.886110e+06, 1.465072e+07, 2.415495e+07, 3.982478e+07, 6.565997e+07, 1.082550e+08, 1.784823e+08, 2.942676e+08, 4.851652e+08, 7.999021e+08, 1.318816e+09, 2.174360e+09, 3.584913e+09, 5.910522e+09, 9.744803e+09, 1.606646e+10, 2.648912e+10, 4.367318e+10, 7.200490e+10, 1.187160e+11, 1.957296e+11, 3.227036e+11, 5.320482e+11, 8.771992e+11, 1.446257e+12, 2.384475e+12, 3.931334e+12, 6.481675e+12, 1.068647e+13, 1.761902e+13, 2.904885e+13, 4.789346e+13, 7.896296e+13, 1.301879e+14, 2.146436e+14, 3.538874e+14, 5.834617e+14, 9.619658e+14, 1.586013e+15, 2.614894e+15, 4.311232e+15, 7.108019e+15, 1.171914e+16, 1.932160e+16, 3.185593e+16, 5.252155e+16, 8.659340e+16, 1.427684e+17, 2.353853e+17, 3.880847e+17, 6.398435e+17, 1.054924e+18, 1.739275e+18, 2.867580e+18, 4.727840e+18, 7.794889e+18, 1.285160e+19, 2.118871e+19, 3.493427e+19, 5.759687e+19, 9.496120e+19, 1.565645e+20, 2.581313e+20, 4.255865e+20, 7.016736e+20, 1.156864e+21, 1.907346e+21, 3.144683e+21, 5.184705e+21, 8.548134e+21, 1.409349e+22, 2.323624e+22, 3.831008e+22, 6.316264e+22, 1.041376e+23, 1.716939e+23, 2.830753e+23, 4.667123e+23, 7.694785e+23, 1.268656e+24, 2.091659e+24, 3.448563e+24, 5.685720e+24, 9.374167e+24, 1.545539e+25, 2.548163e+25, 4.201210e+25, 6.926625e+25, 1.142007e+26, 1.882852e+26, 3.104298e+26, 5.118122e+26, 8.438357e+26, 1.391250e+27, 2.293783e+27, 3.781809e+27 },
       {
1.266417e-14, 1.626111e-14, 2.087968e-14, 2.681004e-14, 3.442477e-14, 4.420228e-14, 5.675685e-14, 7.287724e-14, 9.357623e-14, 1.201543e-13, 1.542811e-13, 1.981009e-13, 2.543666e-13, 3.266131e-13, 4.193796e-13, 5.384941e-13, 6.914400e-13, 8.878265e-13, 1.139992e-12, 1.463779e-12, 1.879529e-12, 2.413363e-12, 3.098819e-12, 3.978963e-12, 5.109089e-12, 6.560200e-12, 8.423464e-12, 1.081594e-11, 1.388794e-11, 1.783247e-11, 2.289735e-11, 2.940078e-11, 3.775135e-11, 4.847369e-11, 6.224145e-11, 7.991960e-11, 1.026188e-10, 1.317651e-10, 1.691898e-10, 2.172440e-10, 2.789468e-10, 3.581748e-10, 4.599056e-10, 5.905304e-10, 7.582560e-10, 9.736201e-10, 1.250153e-09, 1.605228e-09, 2.061154e-09, 2.646574e-09, 3.398268e-09, 4.363462e-09, 5.602796e-09, 7.194133e-09, 9.237450e-09, 1.186112e-08, 1.522998e-08, 1.955568e-08, 2.510999e-08, 3.224187e-08, 4.139938e-08, 5.315785e-08, 6.825604e-08, 8.764248e-08, 1.125352e-07, 1.444980e-07, 1.855391e-07, 2.382370e-07, 3.059023e-07, 3.927863e-07, 5.043477e-07, 6.475952e-07, 8.315287e-07, 1.067704e-06, 1.370959e-06, 1.760346e-06, 2.260329e-06, 2.902320e-06, 3.726653e-06, 4.785118e-06, 6.144212e-06, 7.889324e-06, 1.013009e-05, 1.300730e-05, 1.670170e-05, 2.144541e-05, 2.753645e-05, 3.535750e-05, 4.539993e-05, 5.829466e-05, 7.485183e-05, 9.611165e-05, 1.234098e-04, 1.584613e-04, 2.034684e-04, 2.612586e-04, 3.354626e-04, 4.307425e-04, 5.530844e-04, 7.101744e-04, 9.118820e-04, 1.170880e-03, 1.503439e-03, 1.930454e-03, 2.478752e-03, 3.182781e-03, 4.086772e-03, 5.247518e-03, 6.737947e-03, 8.651695e-03, 1.110900e-02, 1.426423e-02, 1.831564e-02, 2.351775e-02, 3.019738e-02, 3.877421e-02, 4.978707e-02, 6.392786e-02, 8.208500e-02, 1.053992e-01, 1.353353e-01, 1.737739e-01, 2.231302e-01, 2.865048e-01, 3.678795e-01, 4.723666e-01, 6.065307e-01, 7.788008e-01, 1.000000e+00, 1.284025e+00, 1.648721e+00, 2.117000e+00, 2.718282e+00, 3.490343e+00, 4.481689e+00, 5.754602e+00, 7.389056e+00, 9.487736e+00, 1.218249e+01, 1.564263e+01, 2.008554e+01, 2.579034e+01, 3.311545e+01, 4.252108e+01, 5.459815e+01, 7.010541e+01, 9.001714e+01, 1.155843e+02, 1.484132e+02, 1.905663e+02, 2.446919e+02, 3.141906e+02, 4.034288e+02, 5.180128e+02, 6.651416e+02, 8.540587e+02, 1.096633e+03, 1.408105e+03, 1.808042e+03, 2.321573e+03, 2.980958e+03, 3.827626e+03, 4.914769e+03, 6.310688e+03, 8.103084e+03, 1.040457e+04, 1.335973e+04, 1.715423e+04, 2.202646e+04, 2.828254e+04, 3.631550e+04, 4.663003e+04, 5.987414e+04, 7.687992e+04, 9.871577e+04, 1.267536e+05, 1.627548e+05, 2.089813e+05, 2.683373e+05, 3.445519e+05, 4.424134e+05, 5.680701e+05, 7.294164e+05, 9.365891e+05, 1.202604e+06, 1.544174e+06, 1.982759e+06, 2.545913e+06, 3.269017e+06, 4.197502e+06, 5.389698e+06, 6.920510e+06, 8.886110e+06, 1.140999e+07, 1.465072e+07, 1.881190e+07, 2.415495e+07, 3.101557e+07, 3.982478e+07, 5.113603e+07, 6.565997e+07, 8.430906e+07, 1.082550e+08, 1.390022e+08, 1.784823e+08, 2.291758e+08, 2.942676e+08, 3.778470e+08, 4.851652e+08, 6.229644e+08, 7.999021e+08, 1.027095e+09, 1.318816e+09, 1.693393e+09, 2.174360e+09, 2.791933e+09, 3.584913e+09, 4.603119e+09, 5.910522e+09, 7.589260e+09, 9.744803e+09, 1.251258e+10, 1.606646e+10, 2.062975e+10, 2.648912e+10, 3.401270e+10, 4.367318e+10, 5.607747e+10, 7.200490e+10, 9.245612e+10, 1.187160e+11, 1.524344e+11, 1.957296e+11, 2.513218e+11, 3.227036e+11, 4.143596e+11, 5.320482e+11, 6.831635e+11, 8.771992e+11, 1.126346e+12, 1.446257e+12, 1.857031e+12, 2.384475e+12, 3.061726e+12, 3.931334e+12, 5.047933e+12, 6.481675e+12, 8.322635e+12, 1.068647e+13, 1.372171e+13, 1.761902e+13, 2.262327e+13, 2.904885e+13, 3.729946e+13, 4.789346e+13, 6.149642e+13 },
       {
1.125352e-07, 1.275191e-07, 1.444980e-07, 1.637377e-07, 1.855391e-07, 2.102434e-07, 2.382370e-07, 2.699579e-07, 3.059023e-07, 3.466327e-07, 3.927863e-07, 4.450852e-07, 5.043477e-07, 5.715008e-07, 6.475952e-07, 7.338215e-07, 8.315287e-07, 9.422455e-07, 1.067704e-06, 1.209867e-06, 1.370959e-06, 1.553500e-06, 1.760346e-06, 1.994734e-06, 2.260329e-06, 2.561289e-06, 2.902320e-06, 3.288760e-06, 3.726653e-06, 4.222851e-06, 4.785118e-06, 5.422248e-06, 6.144212e-06, 6.962305e-06, 7.889324e-06, 8.939776e-06, 1.013009e-05, 1.147890e-05, 1.300730e-05, 1.473920e-05, 1.670170e-05, 1.892551e-05, 2.144541e-05, 2.430083e-05, 2.753645e-05, 3.120289e-05, 3.535750e-05, 4.006530e-05, 4.539993e-05, 5.144486e-05, 5.829466e-05, 6.605651e-05, 7.485183e-05, 8.481824e-05, 9.611165e-05, 1.089088e-04, 1.234098e-04, 1.398416e-04, 1.584613e-04, 1.795602e-04, 2.034684e-04, 2.305599e-04, 2.612586e-04, 2.960447e-04, 3.354626e-04, 3.801290e-04, 4.307425e-04, 4.880952e-04, 5.530844e-04, 6.267267e-04, 7.101744e-04, 8.047330e-04, 9.118820e-04, 1.033298e-03, 1.170880e-03, 1.326780e-03, 1.503439e-03, 1.703620e-03, 1.930454e-03, 2.187491e-03, 2.478752e-03, 2.808794e-03, 3.182781e-03, 3.606563e-03, 4.086772e-03, 4.630919e-03, 5.247518e-03, 5.946218e-03, 6.737947e-03, 7.635094e-03, 8.651695e-03, 9.803655e-03, 1.110900e-02, 1.258814e-02, 1.426423e-02, 1.616349e-02, 1.831564e-02, 2.075434e-02, 2.351775e-02, 2.664910e-02, 3.019738e-02, 3.421812e-02, 3.877421e-02, 4.393693e-02, 4.978707e-02, 5.641614e-02, 6.392786e-02, 7.243976e-02, 8.208500e-02, 9.301449e-02, 1.053992e-01, 1.194330e-01, 1.353353e-01, 1.533550e-01, 1.737739e-01, 1.969117e-01, 2.231302e-01, 2.528396e-01, 2.865048e-01, 3.246525e-01, 3.678795e-01, 4.168620e-01, 4.723666e-01, 5.352615e-01, 6.065307e-01, 6.872893e-01, 7.788008e-01, 8.824969e-01, 1.000000e+00, 1.133148e+00, 1.284025e+00, 1.454991e+00, 1.648721e+00, 1.868246e+00, 2.117000e+00, 2.398875e+00, 2.718282e+00, 3.080217e+00, 3.490343e+00, 3.955077e+00, 4.481689e+00, 5.078419e+00, 5.754602e+00, 6.520819e+00, 7.389056e+00, 8.372897e+00, 9.487736e+00, 1.075101e+01, 1.218249e+01, 1.380457e+01, 1.564263e+01, 1.772542e+01, 2.008554e+01, 2.275990e+01, 2.579034e+01, 2.922429e+01, 3.311545e+01, 3.752472e+01, 4.252108e+01, 4.818270e+01, 5.459815e+01, 6.186781e+01, 7.010541e+01, 7.943984e+01, 9.001714e+01, 1.020028e+02, 1.155843e+02, 1.309742e+02, 1.484132e+02, 1.681741e+02, 1.905663e+02, 2.159399e+02, 2.446919e+02, 2.772723e+02, 3.141906e+02, 3.560247e+02, 4.034288e+02, 4.571447e+02, 5.180128e+02, 5.869854e+02, 6.651416e+02, 7.537042e+02, 8.540587e+02, 9.677754e+02, 1.096633e+03, 1.242648e+03, 1.408105e+03, 1.595592e+03, 1.808042e+03, 2.048781e+03, 2.321573e+03, 2.630686e+03, 2.980958e+03, 3.377868e+03, 3.827626e+03, 4.337268e+03, 4.914769e+03, 5.569163e+03, 6.310688e+03, 7.150946e+03, 8.103084e+03, 9.181997e+03, 1.040457e+04, 1.178992e+04, 1.335973e+04, 1.513855e+04, 1.715423e+04, 1.943829e+04, 2.202646e+04, 2.495926e+04, 2.828254e+04, 3.204832e+04, 3.631550e+04, 4.115086e+04, 4.663003e+04, 5.283874e+04, 5.987414e+04, 6.784629e+04, 7.687992e+04, 8.711636e+04, 9.871577e+04, 1.118596e+05, 1.267536e+05, 1.436306e+05, 1.627548e+05, 1.844253e+05, 2.089813e+05, 2.368068e+05, 2.683373e+05, 3.040660e+05, 3.445519e+05, 3.904284e+05, 4.424134e+05, 5.013201e+05, 5.680701e+05, 6.437077e+05, 7.294164e+05, 8.265370e+05, 9.365891e+05, 1.061294e+06, 1.202604e+06, 1.362729e+06, 1.544174e+06, 1.749779e+06, 1.982759e+06, 2.246760e+06, 2.545913e+06, 2.884898e+06, 3.269017e+06, 3.704282e+06, 4.197502e+06, 4.756392e+06, 5.389698e+06, 6.107328e+06, 6.920510e+06, 7.841965e+06 },
       {
3.354626e-04, 3.570981e-04, 3.801290e-04, 4.046452e-04, 4.307425e-04, 4.585230e-04, 4.880952e-04, 5.195747e-04, 5.530844e-04, 5.887552e-04, 6.267267e-04, 6.671471e-04, 7.101744e-04, 7.559767e-04, 8.047330e-04, 8.566338e-04, 9.118820e-04, 9.706933e-04, 1.033298e-03, 1.099940e-03, 1.170880e-03, 1.246395e-03, 1.326780e-03, 1.412350e-03, 1.503439e-03, 1.600403e-03, 1.703620e-03, 1.813494e-03, 1.930454e-03, 2.054958e-03, 2.187491e-03, 2.328572e-03, 2.478752e-03, 2.638618e-03, 2.808794e-03, 2.989946e-03, 3.182781e-03, 3.388053e-03, 3.606563e-03, 3.839167e-03, 4.086772e-03, 4.350346e-03, 4.630919e-03, 4.929587e-03, 5.247518e-03, 5.585955e-03, 5.946218e-03, 6.329715e-03, 6.737947e-03, 7.172507e-03, 7.635094e-03, 8.127515e-03, 8.651695e-03, 9.209681e-03, 9.803655e-03, 1.043594e-02, 1.110900e-02, 1.182546e-02, 1.258814e-02, 1.340001e-02, 1.426423e-02, 1.518420e-02, 1.616349e-02, 1.720595e-02, 1.831564e-02, 1.949690e-02, 2.075434e-02, 2.209288e-02, 2.351775e-02, 2.503451e-02, 2.664910e-02, 2.836782e-02, 3.019738e-02, 3.214495e-02, 3.421812e-02, 3.642500e-02, 3.877421e-02, 4.127493e-02, 4.393693e-02, 4.677062e-02, 4.978707e-02, 5.299806e-02, 5.641614e-02, 6.005467e-02, 6.392786e-02, 6.805085e-02, 7.243976e-02, 7.711172e-02, 8.208500e-02, 8.737902e-02, 9.301449e-02, 9.901341e-02, 1.053992e-01, 1.121969e-01, 1.194330e-01, 1.271357e-01, 1.353353e-01, 1.440637e-01, 1.533550e-01, 1.632455e-01, 1.737739e-01, 1.849814e-01, 1.969117e-01, 2.096114e-01, 2.231302e-01, 2.375208e-01, 2.528396e-01, 2.691464e-01, 2.865048e-01, 3.049828e-01, 3.246525e-01, 3.455908e-01, 3.678795e-01, 3.916056e-01, 4.168620e-01, 4.437473e-01, 4.723666e-01, 5.028316e-01, 5.352615e-01, 5.697829e-01, 6.065307e-01, 6.456485e-01, 6.872893e-01, 7.316157e-01, 7.788008e-01, 8.290291e-01, 8.824969e-01, 9.394131e-01, 1.000000e+00, 1.064494e+00, 1.133148e+00, 1.206230e+00, 1.284025e+00, 1.366838e+00, 1.454991e+00, 1.548830e+00, 1.648721e+00, 1.755055e+00, 1.868246e+00, 1.988737e+00, 2.117000e+00, 2.253535e+00, 2.398875e+00, 2.553589e+00, 2.718282e+00, 2.893596e+00, 3.080217e+00, 3.278874e+00, 3.490343e+00, 3.715451e+00, 3.955077e+00, 4.210157e+00, 4.481689e+00, 4.770733e+00, 5.078419e+00, 5.405949e+00, 5.754602e+00, 6.125743e+00, 6.520819e+00, 6.941376e+00, 7.389056e+00, 7.865609e+00, 8.372897e+00, 8.912903e+00, 9.487736e+00, 1.009964e+01, 1.075101e+01, 1.144439e+01, 1.218249e+01, 1.296820e+01, 1.380457e+01, 1.469489e+01, 1.564263e+01, 1.665149e+01, 1.772542e+01, 1.886862e+01, 2.008554e+01, 2.138094e+01, 2.275990e+01, 2.422778e+01, 2.579034e+01, 2.745367e+01, 2.922429e+01, 3.110909e+01, 3.311545e+01, 3.525122e+01, 3.752472e+01, 3.994486e+01, 4.252108e+01, 4.526346e+01, 4.818270e+01, 5.129021e+01, 5.459815e+01, 5.811943e+01, 6.186781e+01, 6.585794e+01, 7.010541e+01, 7.462682e+01, 7.943984e+01, 8.456327e+01, 9.001714e+01, 9.582274e+01, 1.020028e+02, 1.085814e+02, 1.155843e+02, 1.230388e+02, 1.309742e+02, 1.394213e+02, 1.484132e+02, 1.579850e+02, 1.681741e+02, 1.790204e+02, 1.905663e+02, 2.028567e+02, 2.159399e+02, 2.298668e+02, 2.446919e+02, 2.604732e+02, 2.772723e+02, 2.951548e+02, 3.141906e+02, 3.344542e+02, 3.560247e+02, 3.789863e+02, 4.034288e+02, 4.294477e+02, 4.571447e+02, 4.866280e+02, 5.180128e+02, 5.514218e+02, 5.869854e+02, 6.248427e+02, 6.651416e+02, 7.080396e+02, 7.537042e+02, 8.023140e+02, 8.540587e+02, 9.091408e+02, 9.677754e+02, 1.030192e+03, 1.096633e+03, 1.167360e+03, 1.242648e+03, 1.322792e+03, 1.408105e+03, 1.498920e+03, 1.595592e+03, 1.698499e+03, 1.808042e+03, 1.924651e+03, 2.048781e+03, 2.180916e+03, 2.321573e+03, 2.471301e+03, 2.630686e+03, 2.800351e+03 },
       {
1.831564e-02, 1.889704e-02, 1.949690e-02, 2.011579e-02, 2.075434e-02, 2.141315e-02, 2.209288e-02, 2.279418e-02, 2.351775e-02, 2.426428e-02, 2.503451e-02, 2.582919e-02, 2.664910e-02, 2.749503e-02, 2.836782e-02, 2.926831e-02, 3.019738e-02, 3.115595e-02, 3.214495e-02, 3.316534e-02, 3.421812e-02, 3.530432e-02, 3.642500e-02, 3.758125e-02, 3.877421e-02, 4.000504e-02, 4.127493e-02, 4.258514e-02, 4.393693e-02, 4.533164e-02, 4.677062e-02, 4.825528e-02, 4.978707e-02, 5.136748e-02, 5.299806e-02, 5.468040e-02, 5.641614e-02, 5.820698e-02, 6.005467e-02, 6.196101e-02, 6.392786e-02, 6.595715e-02, 6.805085e-02, 7.021102e-02, 7.243976e-02, 7.473924e-02, 7.711172e-02, 7.955951e-02, 8.208500e-02, 8.469066e-02, 8.737902e-02, 9.015273e-02, 9.301449e-02, 9.596708e-02, 9.901341e-02, 1.021564e-01, 1.053992e-01, 1.087450e-01, 1.121969e-01, 1.157584e-01, 1.194330e-01, 1.232242e-01, 1.271357e-01, 1.311715e-01, 1.353353e-01, 1.396313e-01, 1.440637e-01, 1.486367e-01, 1.533550e-01, 1.582230e-01, 1.632455e-01, 1.684275e-01, 1.737739e-01, 1.792901e-01, 1.849814e-01, 1.908533e-01, 1.969117e-01, 2.031623e-01, 2.096114e-01, 2.162652e-01, 2.231302e-01, 2.302131e-01, 2.375208e-01, 2.450605e-01, 2.528396e-01, 2.608656e-01, 2.691464e-01, 2.776900e-01, 2.865048e-01, 2.955994e-01, 3.049828e-01, 3.146640e-01, 3.246525e-01, 3.349580e-01, 3.455908e-01, 3.565610e-01, 3.678795e-01, 3.795572e-01, 3.916056e-01, 4.040365e-01, 4.168620e-01, 4.300947e-01, 4.437473e-01, 4.578333e-01, 4.723666e-01, 4.873611e-01, 5.028316e-01, 5.187932e-01, 5.352615e-01, 5.522525e-01, 5.697829e-01, 5.878696e-01, 6.065307e-01, 6.257840e-01, 6.456485e-01, 6.661436e-01, 6.872893e-01, 7.091062e-01, 7.316157e-01, 7.548396e-01, 7.788008e-01, 8.035226e-01, 8.290291e-01, 8.553454e-01, 8.824969e-01, 9.105104e-01, 9.394131e-01, 9.692333e-01, 1.000000e+00, 1.031743e+00, 1.064494e+00, 1.098285e+00, 1.133148e+00, 1.169118e+00, 1.206230e+00, 1.244520e+00, 1.284025e+00, 1.324785e+00, 1.366838e+00, 1.410226e+00, 1.454991e+00, 1.501178e+00, 1.548830e+00, 1.597996e+00, 1.648721e+00, 1.701057e+00, 1.755055e+00, 1.810766e+00, 1.868246e+00, 1.927550e+00, 1.988737e+00, 2.051867e+00, 2.117000e+00, 2.184201e+00, 2.253535e+00, 2.325070e+00, 2.398875e+00, 2.475024e+00, 2.553589e+00, 2.634649e+00, 2.718282e+00, 2.804569e+00, 2.893596e+00, 2.985448e+00, 3.080217e+00, 3.177993e+00, 3.278874e+00, 3.382956e+00, 3.490343e+00, 3.601138e+00, 3.715451e+00, 3.833392e+00, 3.955077e+00, 4.080624e+00, 4.210157e+00, 4.343802e+00, 4.481689e+00, 4.623953e+00, 4.770733e+00, 4.922173e+00, 5.078419e+00, 5.239625e+00, 5.405949e+00, 5.577552e+00, 5.754602e+00, 5.937274e+00, 6.125743e+00, 6.320195e+00, 6.520819e+00, 6.727812e+00, 6.941376e+00, 7.161719e+00, 7.389056e+00, 7.623610e+00, 7.865609e+00, 8.115291e+00, 8.372897e+00, 8.638681e+00, 8.912903e+00, 9.195829e+00, 9.487736e+00, 9.788909e+00, 1.009964e+01, 1.042024e+01, 1.075101e+01, 1.109229e+01, 1.144439e+01, 1.180768e+01, 1.218249e+01, 1.256921e+01, 1.296820e+01, 1.337985e+01, 1.380457e+01, 1.424278e+01, 1.469489e+01, 1.516136e+01, 1.564263e+01, 1.613918e+01, 1.665149e+01, 1.718007e+01, 1.772542e+01, 1.828809e+01, 1.886862e+01, 1.946757e+01, 2.008554e+01, 2.072312e+01, 2.138094e+01, 2.205965e+01, 2.275990e+01, 2.348237e+01, 2.422778e+01, 2.499685e+01, 2.579034e+01, 2.660901e+01, 2.745367e+01, 2.832515e+01, 2.922429e+01, 3.015196e+01, 3.110909e+01, 3.209660e+01, 3.311545e+01, 3.416665e+01, 3.525122e+01, 3.637021e+01, 3.752472e+01, 3.871589e+01, 3.994486e+01, 4.121284e+01, 4.252108e+01, 4.387085e+01, 4.526346e+01, 4.670027e+01, 4.818270e+01, 4.971218e+01, 5.129021e+01, 5.291834e+01 },
       {
1.353353e-01, 1.374665e-01, 1.396313e-01, 1.418302e-01, 1.440637e-01, 1.463323e-01, 1.486367e-01, 1.509774e-01, 1.533550e-01, 1.557700e-01, 1.582230e-01, 1.607146e-01, 1.632455e-01, 1.658162e-01, 1.684275e-01, 1.710798e-01, 1.737739e-01, 1.765105e-01, 1.792901e-01, 1.821135e-01, 1.849814e-01, 1.878944e-01, 1.908533e-01, 1.938588e-01, 1.969117e-01, 2.000126e-01, 2.031623e-01, 2.063617e-01, 2.096114e-01, 2.129123e-01, 2.162652e-01, 2.196708e-01, 2.231302e-01, 2.266439e-01, 2.302131e-01, 2.338384e-01, 2.375208e-01, 2.412612e-01, 2.450605e-01, 2.489197e-01, 2.528396e-01, 2.568212e-01, 2.608656e-01, 2.649736e-01, 2.691464e-01, 2.733848e-01, 2.776900e-01, 2.820629e-01, 2.865048e-01, 2.910166e-01, 2.955994e-01, 3.002545e-01, 3.049828e-01, 3.097855e-01, 3.146640e-01, 3.196192e-01, 3.246525e-01, 3.297650e-01, 3.349580e-01, 3.402329e-01, 3.455908e-01, 3.510330e-01, 3.565610e-01, 3.621760e-01, 3.678795e-01, 3.736727e-01, 3.795572e-01, 3.855343e-01, 3.916056e-01, 3.977725e-01, 4.040365e-01, 4.103992e-01, 4.168620e-01, 4.234266e-01, 4.300947e-01, 4.368677e-01, 4.437473e-01, 4.507353e-01, 4.578333e-01, 4.650432e-01, 4.723666e-01, 4.798052e-01, 4.873611e-01, 4.950359e-01, 5.028316e-01, 5.107500e-01, 5.187932e-01, 5.269630e-01, 5.352615e-01, 5.436906e-01, 5.522525e-01, 5.609491e-01, 5.697829e-01, 5.787556e-01, 5.878696e-01, 5.971273e-01, 6.065307e-01, 6.160821e-01, 6.257840e-01, 6.356387e-01, 6.456485e-01, 6.558160e-01, 6.661436e-01, 6.766338e-01, 6.872893e-01, 6.981125e-01, 7.091062e-01, 7.202730e-01, 7.316157e-01, 7.431369e-01, 7.548396e-01, 7.667266e-01, 7.788008e-01, 7.910651e-01, 8.035226e-01, 8.161762e-01, 8.290291e-01, 8.420844e-01, 8.553454e-01, 8.688151e-01, 8.824969e-01, 8.963942e-01, 9.105104e-01, 9.248488e-01, 9.394131e-01, 9.542067e-01, 9.692333e-01, 9.844965e-01, 1.000000e+00, 1.015748e+00, 1.031743e+00, 1.047991e+00, 1.064494e+00, 1.081258e+00, 1.098285e+00, 1.115581e+00, 1.133148e+00, 1.150993e+00, 1.169118e+00, 1.187529e+00, 1.206230e+00, 1.225226e+00, 1.244520e+00, 1.264118e+00, 1.284025e+00, 1.304246e+00, 1.324785e+00, 1.345647e+00, 1.366838e+00, 1.388362e+00, 1.410226e+00, 1.432434e+00, 1.454991e+00, 1.477904e+00, 1.501178e+00, 1.524818e+00, 1.548830e+00, 1.573221e+00, 1.597996e+00, 1.623160e+00, 1.648721e+00, 1.674685e+00, 1.701057e+00, 1.727845e+00, 1.755055e+00, 1.782693e+00, 1.810766e+00, 1.839281e+00, 1.868246e+00, 1.897667e+00, 1.927550e+00, 1.957905e+00, 1.988737e+00, 2.020056e+00, 2.051867e+00, 2.084179e+00, 2.117000e+00, 2.150338e+00, 2.184201e+00, 2.218597e+00, 2.253535e+00, 2.289023e+00, 2.325070e+00, 2.361684e+00, 2.398875e+00, 2.436652e+00, 2.475024e+00, 2.514000e+00, 2.553589e+00, 2.593803e+00, 2.634649e+00, 2.676139e+00, 2.718282e+00, 2.761088e+00, 2.804569e+00, 2.848735e+00, 2.893596e+00, 2.939163e+00, 2.985448e+00, 3.032462e+00, 3.080217e+00, 3.128723e+00, 3.177993e+00, 3.228040e+00, 3.278874e+00, 3.330508e+00, 3.382956e+00, 3.436230e+00, 3.490343e+00, 3.545308e+00, 3.601138e+00, 3.657848e+00, 3.715451e+00, 3.773961e+00, 3.833392e+00, 3.893759e+00, 3.955077e+00, 4.017360e+00, 4.080624e+00, 4.144885e+00, 4.210157e+00, 4.276457e+00, 4.343802e+00, 4.412207e+00, 4.481689e+00, 4.552265e+00, 4.623953e+00, 4.696770e+00, 4.770733e+00, 4.845861e+00, 4.922173e+00, 4.999685e+00, 5.078419e+00, 5.158392e+00, 5.239625e+00, 5.322137e+00, 5.405949e+00, 5.491080e+00, 5.577552e+00, 5.665386e+00, 5.754602e+00, 5.845224e+00, 5.937274e+00, 6.030772e+00, 6.125743e+00, 6.222209e+00, 6.320195e+00, 6.419723e+00, 6.520819e+00, 6.623507e+00, 6.727812e+00, 6.833760e+00, 6.941376e+00, 7.050686e+00, 7.161719e+00, 7.274499e+00 },
       {
3.678795e-01, 3.707648e-01, 3.736727e-01, 3.766035e-01, 3.795572e-01, 3.825341e-01, 3.855343e-01, 3.885581e-01, 3.916056e-01, 3.946770e-01, 3.977725e-01, 4.008923e-01, 4.040365e-01, 4.072054e-01, 4.103992e-01, 4.136180e-01, 4.168620e-01, 4.201315e-01, 4.234266e-01, 4.267476e-01, 4.300947e-01, 4.334679e-01, 4.368677e-01, 4.402940e-01, 4.437473e-01, 4.472277e-01, 4.507353e-01, 4.542705e-01, 4.578333e-01, 4.614242e-01, 4.650432e-01, 4.686906e-01, 4.723666e-01, 4.760714e-01, 4.798052e-01, 4.835684e-01, 4.873611e-01, 4.911835e-01, 4.950359e-01, 4.989185e-01, 5.028316e-01, 5.067753e-01, 5.107500e-01, 5.147559e-01, 5.187932e-01, 5.228621e-01, 5.269630e-01, 5.310960e-01, 5.352615e-01, 5.394595e-01, 5.436906e-01, 5.479548e-01, 5.522525e-01, 5.565838e-01, 5.609491e-01, 5.653487e-01, 5.697829e-01, 5.742517e-01, 5.787556e-01, 5.832949e-01, 5.878696e-01, 5.924804e-01, 5.971273e-01, 6.018106e-01, 6.065307e-01, 6.112877e-01, 6.160821e-01, 6.209141e-01, 6.257840e-01, 6.306921e-01, 6.356387e-01, 6.406240e-01, 6.456485e-01, 6.507124e-01, 6.558160e-01, 6.609597e-01, 6.661436e-01, 6.713682e-01, 6.766338e-01, 6.819407e-01, 6.872893e-01, 6.926798e-01, 6.981125e-01, 7.035879e-01, 7.091062e-01, 7.146678e-01, 7.202730e-01, 7.259222e-01, 7.316157e-01, 7.373538e-01, 7.431369e-01, 7.489654e-01, 7.548396e-01, 7.607599e-01, 7.667266e-01, 7.727401e-01, 7.788008e-01, 7.849090e-01, 7.910651e-01, 7.972695e-01, 8.035226e-01, 8.098247e-01, 8.161762e-01, 8.225776e-01, 8.290291e-01, 8.355313e-01, 8.420844e-01, 8.486890e-01, 8.553454e-01, 8.620539e-01, 8.688151e-01, 8.756292e-01, 8.824969e-01, 8.894184e-01, 8.963942e-01, 9.034247e-01, 9.105104e-01, 9.176516e-01, 9.248488e-01, 9.321025e-01, 9.394131e-01, 9.467810e-01, 9.542067e-01, 9.616906e-01, 9.692333e-01, 9.768350e-01, 9.844965e-01, 9.922180e-01, 1.000000e+00, 1.007843e+00, 1.015748e+00, 1.023714e+00, 1.031743e+00, 1.039835e+00, 1.047991e+00, 1.056211e+00, 1.064494e+00, 1.072843e+00, 1.081258e+00, 1.089738e+00, 1.098285e+00, 1.106899e+00, 1.115581e+00, 1.124330e+00, 1.133148e+00, 1.142036e+00, 1.150993e+00, 1.160020e+00, 1.169118e+00, 1.178288e+00, 1.187529e+00, 1.196843e+00, 1.206230e+00, 1.215691e+00, 1.225226e+00, 1.234835e+00, 1.244520e+00, 1.254281e+00, 1.264118e+00, 1.274033e+00, 1.284025e+00, 1.294096e+00, 1.304246e+00, 1.314475e+00, 1.324785e+00, 1.335175e+00, 1.345647e+00, 1.356201e+00, 1.366838e+00, 1.377558e+00, 1.388362e+00, 1.399252e+00, 1.410226e+00, 1.421287e+00, 1.432434e+00, 1.443669e+00, 1.454991e+00, 1.466403e+00, 1.477904e+00, 1.489496e+00, 1.501178e+00, 1.512952e+00, 1.524818e+00, 1.536777e+00, 1.548830e+00, 1.560978e+00, 1.573221e+00, 1.585560e+00, 1.597996e+00, 1.610529e+00, 1.623160e+00, 1.635891e+00, 1.648721e+00, 1.661652e+00, 1.674685e+00, 1.687819e+00, 1.701057e+00, 1.714399e+00, 1.727845e+00, 1.741397e+00, 1.755055e+00, 1.768820e+00, 1.782693e+00, 1.796674e+00, 1.810766e+00, 1.824968e+00, 1.839281e+00, 1.853707e+00, 1.868246e+00, 1.882899e+00, 1.897667e+00, 1.912550e+00, 1.927550e+00, 1.942668e+00, 1.957905e+00, 1.973261e+00, 1.988737e+00, 2.004335e+00, 2.020056e+00, 2.035899e+00, 2.051867e+00, 2.067960e+00, 2.084179e+00, 2.100525e+00, 2.117000e+00, 2.133604e+00, 2.150338e+00, 2.167203e+00, 2.184201e+00, 2.201332e+00, 2.218597e+00, 2.235998e+00, 2.253535e+00, 2.271209e+00, 2.289023e+00, 2.306976e+00, 2.325070e+00, 2.343305e+00, 2.361684e+00, 2.380207e+00, 2.398875e+00, 2.417690e+00, 2.436652e+00, 2.455763e+00, 2.475024e+00, 2.494436e+00, 2.514000e+00, 2.533717e+00, 2.553589e+00, 2.573617e+00, 2.593803e+00, 2.614146e+00, 2.634649e+00, 2.655313e+00, 2.676139e+00, 2.697128e+00 },
};
