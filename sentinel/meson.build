rpicam_app_src += files([
    'sentinel_service.cpp',
])

sentinel_headers = files([
    'sentinel_service.hpp',
])

enable_libcurl = false
libcurl_dep = dependency('libcurl', required: get_option('enable_libcurl'))
if libcurl_dep.found()
    rpicam_app_src += files('alert_service.cpp')
    sentinel_headers += files('alert_service.hpp')
    rpicam_app_dep += libcurl_dep
    cpp_arguments += '-DLIBCURL_PRESENT=1'
    enable_libcurl = true

    assets_dir = meson.project_source_root() / 'assets' / 'email'
    email_assets = files([
        assets_dir / 'alert.html'
    ])
    install_data(email_assets,
             install_dir : get_option('datadir') / 'email-assets')
endif

install_headers(sentinel_headers, subdir: meson.project_name() / 'sentinel')