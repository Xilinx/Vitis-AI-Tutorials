export GPUID=0
export WKDIR=`pwd`

cd $WKDIR
vai_q_caffe test -model float_test.prototxt -weights float.caffemodel -test_iter 500 -gpu $GPUID 2>&1 | tee test_results/float_model_test.txt

#working directory
work_dir=$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

vai_q_caffe quantize            \
          -model ${model_dir}/float_test.prototxt     \
          -weights ${model_dir}/float.caffemodel \
          -gpu $GPUID \
          -keep_fixed_neuron \
          -calib_iter 1000 \
          -test_iter 500 \
          -auto_test \
          -output_dir ${output_dir} 2>&1 | tee test_results/quantized_model_test.txt
