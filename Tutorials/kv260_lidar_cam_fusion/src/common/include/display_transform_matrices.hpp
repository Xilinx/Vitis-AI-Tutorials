/*
 * Copyright 2022 Xilinx, Inc.
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
 
#ifndef _display_transform_matrices_hpp_
#define _display_transform_matrices_hpp_

#include <vitis/ai/pointpillars.hpp>
  
class display_transform_matrices
{
	private:
		static constexpr float P2 [4][4] = 
		{
			{ 721.54,   0,    609.56,  44.857  },
			{   0,    721.54, 172.854, 0.21638 },
			{   0,      0,      1,     0.002746},
			{   0,      0,      0,     1       }
		};

		static constexpr float rect [4][4]
		{
			{  0.999924, 0.009838,    -0.007445,   0},
			{ -0.00987,  0.99994,     -0.00427846, 0},
			{  0.007403, 0.004351614,  0.999963,   0},
			{  0,        0,            0,          1}
		};

		static constexpr float Trv2c [4][4]
		{
			{ 0.007534, -0.99997 ,  -0.0006166, -0.00407},
			{ 0.0148,    0.000728,  -0.99989,   -0.07632},
			{ 0.99986,   0.0075238,  0.0148,    -0.27178},
			{ 0,         0,          0,          1      }
		};
		
		inline static vitis::ai::DISPLAY_PARAM _display_param;
		
	public:
		display_transform_matrices();
		vitis::ai::DISPLAY_PARAM& get_display_data();
		~display_transform_matrices();
};

#endif
