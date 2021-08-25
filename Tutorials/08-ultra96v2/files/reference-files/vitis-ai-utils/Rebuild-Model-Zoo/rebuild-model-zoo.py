###
# Copyright 2020 Xilinx
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import os, sys, argparse, shutil
import json
import subprocess

def exit_app():
    print("\n")
    exit()

def cf_comp(path="", model_name="", net_name=""):

    options_c = { 'mode' : 'normal'}

    if (path=="" or model_name=="" or net_name==""):
        print("\t-- Compiler parameters missing, exiting")
        exit_app()

    if os.path.exists('./compiled_output/' + model_name):
        print("\t-- Removing previously built model")
        shutil.rmtree('./compiled_output/' + model_name)

    cmd = "vai_c_caffe --prototxt " + path + model_name + "/compiler/deploy.prototxt --caffemodel "
    cmd = cmd + path + model_name + "/compiler/deploy.caffemodel --arch ./custom_arch_files/cust.json --output_dir ./compiled_output/"
    cmd = cmd + model_name + " --net_name " + net_name + " --options \"" + str(options_c) + "\""

    os.system(cmd)
    # while True:
    #     output = cmproc.stdout.readline()
    #     if output == '' and cmproc.poll() is not None:
    #         break
    #     if output:
    #         print(output.strip())
    #     if cmproc.poll():
    #         break

clarg_parser = argparse.ArgumentParser()

clarg_parser.add_argument('-z',
    '--model_zoo_dir',
    action='store',
    type=str,
    metavar='<model zoo location>',
    required=True)

clarg_parser.add_argument('-n',
    '--pfm_name',
    action='store',
    type=str,
    metavar='<custom pfm name: %.dcf>',
    required=True)

clarg_parser.add_argument('-m',
    '--model_name',
    action='store',
    type=str,
    metavar='<model name to compile or all-models>')

clargs = clarg_parser.parse_args()

print("\n-----Running model-zoo recompile utility-----\n")

if not os.path.exists("./custom_arch_files/" + str(clargs.pfm_name) + ".dcf"):
    print("\t-- Architecture file ./custom_arch_files/" + clargs.pfm_name +".dcf not found, exiting")
    exit_app()
else:
    print("\t-- Architecture file ./custom_arch_files/" + clargs.pfm_name +".dcf found, generating arch json")
    with open("./custom_arch_files/cust.json", "w") as json_f:
        data={}
        data['target']="dpuv2"
        data['dcf']="./custom_arch_files/" + str(clargs.pfm_name) + ".dcf"
        data['cpu_arch']="arm64"
        json.dump(data,json_f)
    print("\t-- json file generated")

if not os.path.exists(str(clargs.model_zoo_dir)):
    print("\t-- given Model Zoo directory does not exist, exiting")
    exit_app()
else:
    print("\t-- given Model Zoo directory exists, found models:")
    for model in os.scandir(clargs.model_zoo_dir):
        print("\t\t* : " + str(model.name))

if not os.path.exists('compiled_output'):
    print("\t-- Creating directory for compiled model outputs")
    os.mkdir("./compiled_output")

if clargs.model_name != None:
    print("\t-- Building selected model: " + clargs.model_name)
    cf_comp(clargs.model_zoo_dir, clargs.model_name, clargs.model_name.split('_')[1])
else:
    #only caffe models for now...
    print("\t-- Building all models!")
    for model in os.scandir(clargs.model_zoo_dir):
        print("\t\n-- Building : " + str(model.name) + "...")
        if model.name.split('_')[0] == "cf":
            cf_comp(clargs.model_zoo_dir, model.name, model.name.split('_')[1])

exit_app()
