"""GriffithDemo.xcodeproj/project.pbxproj を生成する(xcodegen 不要・標準ライブラリのみ)。

gen_model.py が最後に呼ぶ。Models/<name>/ を走査した結果を渡すと、
  - アプリ本体(GriffithDemo/*.swift, GdModel.h, GdRun.c, bridging header)
  - ランタイム(runtime/c/src/mdl_kernels.c を **相対パスで参照**。コピーしない)
  - モデルごとの <name>_unit.c(コンパイル)と <name>_samples.json(Resources)
  - Models/gd_registry.c
を持つプロジェクトを書く。単独実行は `python gen_xcodeproj.py`(Models/ を自動走査)。

ID は内容から決定的に作る(再生成しても差分が出ない)。
"""
# ruff: noqa: E501  (pbxproj テンプレートの行はフォーマット上折り返せない)

from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = 'GriffithDemo'
BUNDLE_ID = 'com.example.griffithdemo'


def _id(key: str) -> str:
    return hashlib.md5(f'{PROJ}:{key}'.encode()).hexdigest()[:24].upper()


def _q(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _ftype(path: str) -> str:
    return {'.swift': 'sourcecode.swift', '.c': 'sourcecode.c.c', '.h': 'sourcecode.c.h',
            '.json': 'text.json'}[Path(path).suffix]


def build_pbxproj(models: list[str]) -> str:
    # (key, path, sourceTree, group, role)  role: source | resource | none
    files: list[tuple[str, str, str, str, str]] = []
    app = [('GriffithDemoApp.swift', 'source'), ('ContentView.swift', 'source'),
           ('Inference.swift', 'source'), ('GdModel.h', 'none'), ('GdRun.c', 'source'),
           ('GriffithDemo-Bridging-Header.h', 'none')]
    for fn, role in app:
        files.append((f'app/{fn}', fn, '<group>', 'app', role))
    files.append(('rt/mdl_kernels.c', '../../runtime/c/src/mdl_kernels.c', 'SOURCE_ROOT',
                  'runtime', 'source'))
    files.append(('rt/mdl_kernels.h', '../../runtime/c/include/mdl_kernels.h', 'SOURCE_ROOT',
                  'runtime', 'none'))
    files.append(('rt/mdl_simd.h', '../../runtime/c/include/mdl_simd.h', 'SOURCE_ROOT',
                  'runtime', 'none'))
    files.append(('models/gd_registry.c', 'gd_registry.c', '<group>', 'models', 'source'))
    for n in models:
        files.append((f'm/{n}/unit', f'{n}_unit.c', '<group>', f'm/{n}', 'source'))
        files.append((f'm/{n}/model.c', 'model.c', '<group>', f'm/{n}', 'none'))
        files.append((f'm/{n}/model.h', 'model.h', '<group>', f'm/{n}', 'none'))
        files.append((f'm/{n}/samples', f'{n}_samples.json', '<group>', f'm/{n}', 'resource'))

    product = _id('product')
    target = _id('target')
    project = _id('project')
    main_group = _id('group/main')
    products_group = _id('group/products')
    cfg_proj, cfg_tgt = _id('cfglist/project'), _id('cfglist/target')
    ph_src, ph_fw, ph_res = _id('phase/src'), _id('phase/fw'), _id('phase/res')

    def group(gid: str, name: str, path: str | None, children: list[str]) -> str:
        p = f'\t\t\tpath = {_q(path)};\n' if path else ''
        n = f'\t\t\tname = {_q(name)};\n' if name != path else ''
        ch = ''.join(f'\t\t\t\t{c},\n' for c in children)
        return (f'\t\t{gid} /* {name} */ = {{\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (\n{ch}'
                f'\t\t\t);\n{p}{n}\t\t\tsourceTree = "<group>";\n\t\t}};\n')

    def children_of(gkey: str) -> list[str]:
        return [_id('file/' + k) for k, _, _, g, _ in files if g == gkey]

    groups = []
    gid_app, gid_rt, gid_models = _id('group/app'), _id('group/runtime'), _id('group/models')
    groups.append(group(main_group, PROJ, None,
                        [gid_app, gid_rt, gid_models, products_group]))
    groups.append(group(gid_app, 'GriffithDemo', 'GriffithDemo', children_of('app')))
    groups.append(group(gid_rt, 'runtime-c', None, children_of('runtime')))
    model_groups = [_id(f'group/m/{n}') for n in models]
    groups.append(group(gid_models, 'Models', 'Models',
                        children_of('models') + model_groups))
    for n, gid in zip(models, model_groups, strict=False):
        groups.append(group(gid, n, n, children_of(f'm/{n}')))
    groups.append(group(products_group, 'Products', None, [product]))

    refs, bfiles, src_ids, res_ids = [], [], [], []
    for k, path, tree, _g, role in files:
        fid = _id('file/' + k)
        name = Path(path).name
        nm = f'name = {_q(name)}; ' if name != path else ''
        refs.append(f'\t\t{fid} /* {name} */ = {{isa = PBXFileReference; '
                    f'lastKnownFileType = {_ftype(path)}; {nm}path = {_q(path)}; '
                    f'sourceTree = {_q(tree) if tree != "<group>" else chr(34) + "<group>" + chr(34)}; }};')
        if role in ('source', 'resource'):
            bid = _id('build/' + k)
            phase = 'Sources' if role == 'source' else 'Resources'
            bfiles.append(f'\t\t{bid} /* {name} in {phase} */ = {{isa = PBXBuildFile; '
                          f'fileRef = {fid} /* {name} */; }};')
            (src_ids if role == 'source' else res_ids).append((bid, name, phase))
    refs.append(f'\t\t{product} /* {PROJ}.app */ = {{isa = PBXFileReference; '
                f'explicitFileType = wrapper.application; includeInIndex = 0; '
                f'path = {PROJ}.app; sourceTree = BUILT_PRODUCTS_DIR; }};')

    def phase_files(items):
        return ''.join(f'\t\t\t\t{b} /* {n} in {p} */,\n' for b, n, p in items)

    common = {
        'ALWAYS_SEARCH_USER_PATHS': 'NO',
        'CLANG_ENABLE_MODULES': 'YES',
        'CLANG_ENABLE_OBJC_ARC': 'YES',
        'CLANG_WARN_DOCUMENTATION_COMMENTS': 'NO',
        'COPY_PHASE_STRIP': 'NO',
        'GCC_C_LANGUAGE_STANDARD': 'gnu11',
        'IPHONEOS_DEPLOYMENT_TARGET': '16.0',
        'SDKROOT': 'iphoneos',
        'SWIFT_VERSION': '5.0',
    }
    proj_dbg = dict(common, **{
        'DEBUG_INFORMATION_FORMAT': 'dwarf', 'ENABLE_TESTABILITY': 'YES',
        'GCC_DYNAMIC_NO_PIC': 'NO', 'ONLY_ACTIVE_ARCH': 'YES',
        'SWIFT_ACTIVE_COMPILATION_CONDITIONS': 'DEBUG',
        'SWIFT_OPTIMIZATION_LEVEL': '"-Onone"',
        # C(生成コード+ランタイム)は Debug でも -O3: レイテンシ計測を Debug で歪めない
        'GCC_OPTIMIZATION_LEVEL': '3',
    })
    proj_rel = dict(common, **{
        'DEBUG_INFORMATION_FORMAT': '"dwarf-with-dsym"', 'ENABLE_NS_ASSERTIONS': 'NO',
        'SWIFT_COMPILATION_MODE': 'wholemodule', 'VALIDATE_PRODUCT': 'YES',
        'GCC_OPTIMIZATION_LEVEL': '3',
    })
    tgt = {
        'ASSETCATALOG_COMPILER_GENERATE_SWIFT_ASSET_SYMBOL_EXTENSIONS': 'NO',
        'CODE_SIGN_STYLE': 'Automatic',
        'CURRENT_PROJECT_VERSION': '1',
        'DEVELOPMENT_TEAM': '""',
        'ENABLE_PREVIEWS': 'NO',
        'GENERATE_INFOPLIST_FILE': 'YES',
        'HEADER_SEARCH_PATHS': '("$(SRCROOT)/../../runtime/c/include", "$(SRCROOT)/GriffithDemo")',
        'INFOPLIST_KEY_CFBundleDisplayName': '"Griffith Demo"',
        'INFOPLIST_KEY_UIApplicationSceneManifest_Generation': 'YES',
        'INFOPLIST_KEY_UILaunchScreen_Generation': 'YES',
        'INFOPLIST_KEY_UISupportedInterfaceOrientations_iPhone': '"UIInterfaceOrientationPortrait"',
        'LD_RUNPATH_SEARCH_PATHS': '("$(inherited)", "@executable_path/Frameworks")',
        'MARKETING_VERSION': '1.0',
        'PRODUCT_BUNDLE_IDENTIFIER': BUNDLE_ID,
        'PRODUCT_NAME': '"$(TARGET_NAME)"',
        'SWIFT_EMIT_LOC_STRINGS': 'YES',
        'SWIFT_OBJC_BRIDGING_HEADER': '"GriffithDemo/GriffithDemo-Bridging-Header.h"',
        'TARGETED_DEVICE_FAMILY': '1',
    }

    def settings(d: dict) -> str:
        return ''.join(f'\t\t\t\t{k} = {v};\n' for k, v in sorted(d.items()))

    def cfg(cid: str, name: str, d: dict) -> str:
        return (f'\t\t{cid} /* {name} */ = {{\n\t\t\tisa = XCBuildConfiguration;\n'
                f'\t\t\tbuildSettings = {{\n{settings(d)}\t\t\t}};\n\t\t\tname = {name};\n\t\t}};\n')

    cids = {k: _id(f'cfg/{k}') for k in ('pd', 'pr', 'td', 'tr')}

    return f'''// !$*UTF8*$!
{{
\tarchiveVersion = 1;
\tclasses = {{
\t}};
\tobjectVersion = 56;
\tobjects = {{

/* Begin PBXBuildFile section */
{chr(10).join(bfiles)}
/* End PBXBuildFile section */

/* Begin PBXFileReference section */
{chr(10).join(refs)}
/* End PBXFileReference section */

/* Begin PBXFrameworksBuildPhase section */
\t\t{ph_fw} /* Frameworks */ = {{
\t\t\tisa = PBXFrameworksBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXFrameworksBuildPhase section */

/* Begin PBXGroup section */
{''.join(groups)}/* End PBXGroup section */

/* Begin PBXNativeTarget section */
\t\t{target} /* {PROJ} */ = {{
\t\t\tisa = PBXNativeTarget;
\t\t\tbuildConfigurationList = {cfg_tgt} /* Build configuration list for PBXNativeTarget "{PROJ}" */;
\t\t\tbuildPhases = (
\t\t\t\t{ph_src} /* Sources */,
\t\t\t\t{ph_fw} /* Frameworks */,
\t\t\t\t{ph_res} /* Resources */,
\t\t\t);
\t\t\tbuildRules = (
\t\t\t);
\t\t\tdependencies = (
\t\t\t);
\t\t\tname = {PROJ};
\t\t\tproductName = {PROJ};
\t\t\tproductReference = {product} /* {PROJ}.app */;
\t\t\tproductType = "com.apple.product-type.application";
\t\t}};
/* End PBXNativeTarget section */

/* Begin PBXProject section */
\t\t{project} /* Project object */ = {{
\t\t\tisa = PBXProject;
\t\t\tattributes = {{
\t\t\t\tBuildIndependentTargetsInParallel = 1;
\t\t\t\tLastSwiftUpdateCheck = 1600;
\t\t\t\tLastUpgradeCheck = 1600;
\t\t\t\tTargetAttributes = {{
\t\t\t\t\t{target} = {{
\t\t\t\t\t\tCreatedOnToolsVersion = 16.0;
\t\t\t\t\t}};
\t\t\t\t}};
\t\t\t}};
\t\t\tbuildConfigurationList = {cfg_proj} /* Build configuration list for PBXProject "{PROJ}" */;
\t\t\tcompatibilityVersion = "Xcode 14.0";
\t\t\tdevelopmentRegion = en;
\t\t\thasScannedForEncodings = 0;
\t\t\tknownRegions = (
\t\t\t\ten,
\t\t\t\tBase,
\t\t\t);
\t\t\tmainGroup = {main_group};
\t\t\tproductRefGroup = {products_group} /* Products */;
\t\t\tprojectDirPath = "";
\t\t\tprojectRoot = "";
\t\t\ttargets = (
\t\t\t\t{target} /* {PROJ} */,
\t\t\t);
\t\t}};
/* End PBXProject section */

/* Begin PBXResourcesBuildPhase section */
\t\t{ph_res} /* Resources */ = {{
\t\t\tisa = PBXResourcesBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
{phase_files(res_ids)}\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXResourcesBuildPhase section */

/* Begin PBXSourcesBuildPhase section */
\t\t{ph_src} /* Sources */ = {{
\t\t\tisa = PBXSourcesBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
{phase_files(src_ids)}\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXSourcesBuildPhase section */

/* Begin XCBuildConfiguration section */
{cfg(cids['pd'], 'Debug', proj_dbg)}{cfg(cids['pr'], 'Release', proj_rel)}{cfg(cids['td'], 'Debug', tgt)}{cfg(cids['tr'], 'Release', tgt)}/* End XCBuildConfiguration section */

/* Begin XCConfigurationList section */
\t\t{cfg_proj} /* Build configuration list for PBXProject "{PROJ}" */ = {{
\t\t\tisa = XCConfigurationList;
\t\t\tbuildConfigurations = (
\t\t\t\t{cids['pd']} /* Debug */,
\t\t\t\t{cids['pr']} /* Release */,
\t\t\t);
\t\t\tdefaultConfigurationIsVisible = 0;
\t\t\tdefaultConfigurationName = Release;
\t\t}};
\t\t{cfg_tgt} /* Build configuration list for PBXNativeTarget "{PROJ}" */ = {{
\t\t\tisa = XCConfigurationList;
\t\t\tbuildConfigurations = (
\t\t\t\t{cids['td']} /* Debug */,
\t\t\t\t{cids['tr']} /* Release */,
\t\t\t);
\t\t\tdefaultConfigurationIsVisible = 0;
\t\t\tdefaultConfigurationName = Release;
\t\t}};
/* End XCConfigurationList section */
\t}};
\trootObject = {project} /* Project object */;
}}
'''


def write_project(models: list[str]) -> Path:
    proj_dir = HERE / f'{PROJ}.xcodeproj'
    proj_dir.mkdir(exist_ok=True)
    out = proj_dir / 'project.pbxproj'
    out.write_text(build_pbxproj(models), encoding='utf-8')
    return out


if __name__ == '__main__':
    names = sorted(p.name for p in (HERE / 'Models').iterdir()
                   if p.is_dir() and (p / f'{p.name}_unit.c').exists())
    print(write_project(names))
