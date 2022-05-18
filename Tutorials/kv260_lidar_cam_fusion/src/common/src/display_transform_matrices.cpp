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
 
#include "../include/display_transform_matrices.hpp"

display_transform_matrices::display_transform_matrices()
{

	for (int ii = 0; ii < 4; ii++)
	{
		std::vector<float> P2_row;
		std::vector<float> rect_row;
		std::vector<float> Trv2c_row;
		for (int jj = 0; jj < 4; jj++)
		{
			P2_row.push_back(P2[ii][jj]);
			rect_row.push_back(rect[ii][jj]);
			Trv2c_row.push_back(Trv2c[ii][jj]);
		}
		_display_param.P2.emplace_back(P2_row);
		_display_param.rect.emplace_back(rect_row);
		_display_param.Trv2c.emplace_back(Trv2c_row);
	}

	_display_param.p2rect.resize(4);
	for(int i=0; i<4; i++)
	{
		_display_param.p2rect[i].resize(4);
		for(int j=0; j<4; j++)
		{
			for(int k=0; k<4; k++)
			{
				_display_param.p2rect[i][j] += _display_param.P2[i][k]*_display_param.rect[k][j];
			}
		}
	}
};

vitis::ai::DISPLAY_PARAM& display_transform_matrices::get_display_data()
{
	return _display_param;
}

display_transform_matrices::~display_transform_matrices()
{

}

