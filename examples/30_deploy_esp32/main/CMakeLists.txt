# モデル置き場。既定は main/model/(gen_model.py の既定出力)。
# 別モデルを別 build dir で同時に持ちたいときは idf.py -DMODEL_DIR=/abs/path で差し替える。
if(NOT DEFINED MODEL_DIR)
    set(MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/model")
endif()

if(NOT EXISTS "${MODEL_DIR}/model.c")
    message(FATAL_ERROR
        "${MODEL_DIR}/model.c が無い。先に生成すること:\n"
        "  uv run python examples/30_deploy_esp32/gen_model.py [--bundle B --samples S --out DIR]")
endif()

idf_component_register(
    SRCS main.c "${MODEL_DIR}/model.c" "${MODEL_DIR}/mdl_kernels.c"
    INCLUDE_DIRS . "${MODEL_DIR}"
    REQUIRES esp_timer esp_system freertos)

# 既定コンテキスト(static)を消し、作業RAMは main.c の static work[] だけにする
target_compile_definitions(${COMPONENT_LIB} PRIVATE MDL_NO_DEFAULT_CTX)
# 生成コードの警告で -Werror 落ちしないように(ランタイムはホストで別途検証済み)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error)
