import yaml


class Settings:
    def __init__(
            self,
            config_file,
            selected_theme="Default",
            server_name="",
            server_port=0,
            server_share=False,
            output_image_format="png",
            output_video_format="mp4",
            output_video_codec="libx264",
            video_quality=14,
            clear_output=True,
            max_threads=10,
            memory_limit=0,
            provider="cuda",
            force_cpu=False,
            output_template="{file}_{time}",
            use_os_temp_folder=False,
            output_show_video=True,
            launch_browser=True
    ):
        self.config_file = config_file

        # Initialize with provided or default values
        self.selected_theme = selected_theme
        self.server_name = server_name
        self.server_port = server_port
        self.server_share = server_share
        self.output_image_format = output_image_format
        self.output_video_format = output_video_format
        self.output_video_codec = output_video_codec
        self.video_quality = video_quality
        self.clear_output = clear_output
        self.max_threads = max_threads
        self.memory_limit = memory_limit
        self.provider = provider
        self.force_cpu = force_cpu
        self.output_template = output_template
        self.use_os_temp_folder = use_os_temp_folder
        self.output_show_video = output_show_video
        self.launch_browser = launch_browser

        # Load from config file, overriding defaults if present
        self.load()

    def default_get(_, data, name, default):
        value = default
        try:
            value = data.get(name, default)
        except:
            pass
        return value

    def load(self):
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except:
            data = None

        if data:
            self.selected_theme = self.default_get(data, 'selected_theme', self.selected_theme)
            self.server_name = self.default_get(data, 'server_name', self.server_name)
            self.server_port = self.default_get(data, 'server_port', self.server_port)
            self.server_share = self.default_get(data, 'server_share', self.server_share)
            self.output_image_format = self.default_get(data, 'output_image_format', self.output_image_format)
            self.output_video_format = self.default_get(data, 'output_video_format', self.output_video_format)
            self.output_video_codec = self.default_get(data, 'output_video_codec', self.output_video_codec)
            self.video_quality = self.default_get(data, 'video_quality', self.video_quality)
            self.clear_output = self.default_get(data, 'clear_output', self.clear_output)
            self.max_threads = self.default_get(data, 'max_threads', self.max_threads)
            self.memory_limit = self.default_get(data, 'memory_limit', self.memory_limit)
            self.provider = self.default_get(data, 'provider', self.provider)
            self.force_cpu = self.default_get(data, 'force_cpu', self.force_cpu)
            self.output_template = self.default_get(data, 'output_template', self.output_template)
            self.use_os_temp_folder = self.default_get(data, 'use_os_temp_folder', self.use_os_temp_folder)
            self.output_show_video = self.default_get(data, 'output_show_video', self.output_show_video)
            self.launch_browser = self.default_get(data, 'launch_browser', self.launch_browser)

    def save(self):
        data = {
            'selected_theme': self.selected_theme,
            'server_name': self.server_name,
            'server_port': self.server_port,
            'server_share': self.server_share,
            'output_image_format': self.output_image_format,
            'output_video_format': self.output_video_format,
            'output_video_codec': self.output_video_codec,
            'video_quality': self.video_quality,
            'clear_output': self.clear_output,
            'max_threads': self.max_threads,
            'memory_limit': self.memory_limit,
            'provider': self.provider,
            'force_cpu': self.force_cpu,
            'output_template': self.output_template,
            'use_os_temp_folder': self.use_os_temp_folder,
            'output_show_video': self.output_show_video,
            'launch_browser': self.launch_browser
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(data, f)
