rpicam_app_src += files([
    'web_server.cpp',
    'mongoose_wrapper.c',
    'local_handler.cpp'
])

web_headers = files([
    'web_server.hpp',
    'local_handler.hpp'
])

install_subdir('static', install_dir: '/var/www')
install_headers(web_headers, subdir: meson.project_name() / 'web')