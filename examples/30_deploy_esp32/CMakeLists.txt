# ESP32 実機デモ: デプロイキット(codegen-c)の生成モデルを載せる ESP-IDF プロジェクト。
# main/model/ は gen_model.py が生成する(コミットしない)。README.md 参照。
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(deploy_esp32_demo)
