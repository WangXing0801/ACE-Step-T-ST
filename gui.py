import os
import sys
import subprocess
import shutil
from pathlib import Path
import re

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QFileDialog, QLineEdit, QFormLayout, QMessageBox, QGroupBox, QGridLayout
)
from PyQt5.QtCore import QProcess, Qt, QProcessEnvironment, QByteArray
from PyQt5.QtGui import QFont

# 默认参数配置
DEFAULT_PARAMS = {
    'batch_size': 4,
    'num_workers': 0,
    'tag_dropout': 0.5,
    'learning_rate': 1e-4,
    'max_steps': 2000,
    'precision': "bf16-mixed",
    'save_every_n_train_steps': 100,
}

class ACEStepTrainerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ACE-Step 模型训练 GUI")
        self.setGeometry(100, 100, 1100, 800)
        self.base_dir = Path(__file__).parent.resolve()
        self.audio_name = ""
        self.audio_folder = ""
        self.params = DEFAULT_PARAMS.copy()
        self.initUI()
        self.setupReset()
        self.check_python_environment()

    def initUI(self):
        main_layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("ACE-Step 模型训练操作面板    by 圣天制作")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 创建主内容布局
        content_layout = QHBoxLayout()
        
        # 左侧控制面板
        left_panel = QVBoxLayout()
        
        # 训练准备阶段
        prep_group = QGroupBox("📋 训练准备阶段")
        prep_layout = QGridLayout()
        
        # 按钮样式
        button_style = """
            QPushButton {
                padding: 8px;
                font-weight: bold;
            }
        """
        
        self.upload_btn = QPushButton("1. 上传音频文件")
        self.upload_btn.setStyleSheet(button_style)
        self.upload_btn.clicked.connect(self.upload_audio)
        prep_layout.addWidget(self.upload_btn, 0, 0, 1, 2)

        self.gen_prompt_btn = QPushButton("2. 生成提示词")
        self.gen_prompt_btn.setStyleSheet(button_style)
        self.gen_prompt_btn.clicked.connect(self.generate_prompt)
        prep_layout.addWidget(self.gen_prompt_btn, 1, 0)

        self.gen_lyrics_btn = QPushButton("3. 生成歌词")
        self.gen_lyrics_btn.setStyleSheet(button_style)
        self.gen_lyrics_btn.clicked.connect(self.generate_lyrics)
        prep_layout.addWidget(self.gen_lyrics_btn, 1, 1)

        self.create_dataset_btn = QPushButton("4. 创建文件名数据集")
        self.create_dataset_btn.setStyleSheet(button_style)
        self.create_dataset_btn.clicked.connect(self.create_dataset)
        prep_layout.addWidget(self.create_dataset_btn, 2, 0)

        self.preprocess_btn = QPushButton("5. 音频预处理")
        self.preprocess_btn.setStyleSheet(button_style)
        self.preprocess_btn.clicked.connect(self.preprocess_audio)
        prep_layout.addWidget(self.preprocess_btn, 2, 1)

        prep_group.setLayout(prep_layout)
        left_panel.addWidget(prep_group)

        # 训练执行阶段
        train_group = QGroupBox("🏋️ 训练执行阶段")
        train_layout = QVBoxLayout()
        
        self.train_btn = QPushButton("6. 开始训练")
        self.train_btn.setStyleSheet(button_style)
        self.train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(self.train_btn)
        
        train_group.setLayout(train_layout)
        left_panel.addWidget(train_group)

        # 工具按钮区域
        tools_group = QGroupBox("🛠️ 工具")
        tools_layout = QHBoxLayout()
        
        self.tensorboard_btn = QPushButton("启动 TensorBoard")
        self.tensorboard_btn.clicked.connect(self.start_tensorboard)
        tools_layout.addWidget(self.tensorboard_btn)

        self.reset_btn = QPushButton("重置所有")
        self.reset_btn.clicked.connect(self.reset_all)
        tools_layout.addWidget(self.reset_btn)
        
        tools_group.setLayout(tools_layout)
        left_panel.addWidget(tools_group)

        # 参数设置区域
        form_layout = QFormLayout()
        self.batch_size_edit = self.create_param_input(form_layout, 'batch_size')
        self.num_workers_edit = self.create_param_input(form_layout, 'num_workers')
        self.tag_dropout_edit = self.create_param_input(form_layout, 'tag_dropout')
        self.learning_rate_edit = self.create_param_input(form_layout, 'learning_rate')
        self.max_steps_edit = self.create_param_input(form_layout, 'max_steps')
        self.precision_edit = self.create_param_input(form_layout, 'precision')
        self.save_every_n_steps_edit = self.create_param_input(form_layout, 'save_every_n_train_steps')

        params_group = QGroupBox("⚙️ 训练参数设置")
        params_group.setLayout(form_layout)
        left_panel.addWidget(params_group)
        
        # 添加弹簧以改善布局
        left_panel.addStretch()

        # 右侧显示区域
        right_panel = QVBoxLayout()
        
        # 提示词和歌词显示区域
        prompt_lyrics_layout = QHBoxLayout()
        
        # 提示词显示框
        prompt_group = QGroupBox("📝 提示词内容 ({音频名}_prompt.txt)")
        prompt_layout = QVBoxLayout()
        self.prompt_display = QTextEdit()
        self.prompt_display.setReadOnly(True)
        prompt_layout.addWidget(self.prompt_display)
        prompt_group.setLayout(prompt_layout)
        prompt_lyrics_layout.addWidget(prompt_group)

        # 歌词显示框
        lyrics_group = QGroupBox("🎵 歌词内容 ({音频名}_lyrics.txt)")
        lyrics_layout = QVBoxLayout()
        self.lyrics_display = QTextEdit()
        self.lyrics_display.setReadOnly(True)
        lyrics_layout.addWidget(self.lyrics_display)
        lyrics_group.setLayout(lyrics_layout)
        prompt_lyrics_layout.addWidget(lyrics_group)
        
        right_panel.addLayout(prompt_lyrics_layout)

        # 日志框
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        log_group = QGroupBox("📋 运行日志")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_box)
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)

        # 设置左右面板比例
        content_layout.addLayout(left_panel, 1)
        content_layout.addLayout(right_panel, 2)
        
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

    def create_param_input(self, layout, key):
        line_edit = QLineEdit(str(self.params[key]))
        line_edit.setObjectName(key)
        layout.addRow(QLabel(f"{key}:"), line_edit)
        return line_edit

    def setupReset(self):
        self.reset_values = {k: v for k, v in self.params.items()}

    def update_params(self):
        try:
            self.params['batch_size'] = int(self.batch_size_edit.text())
            self.params['num_workers'] = int(self.num_workers_edit.text())
            self.params['tag_dropout'] = float(self.tag_dropout_edit.text())
            self.params['learning_rate'] = float(self.learning_rate_edit.text())
            self.params['max_steps'] = int(self.max_steps_edit.text())
            self.params['precision'] = self.precision_edit.text()
            self.params['save_every_n_train_steps'] = int(self.save_every_n_steps_edit.text())
        except ValueError:
            QMessageBox.warning(self, "参数错误", "请确保所有参数格式正确！")
            return False
        return True

    def log_output(self, output):
        """处理命令行输出，兼容不同编码"""
        try:
            # 如果是 QByteArray 对象，转换为 bytes
            if isinstance(output, QByteArray):
                byte_data = bytes(output)
            else:
                byte_data = output if isinstance(output, bytes) else str(output).encode('utf-8', errors='ignore')
            
            # 尝试多种编码解码
            for encoding in ['utf-8', 'utf-16', 'gbk', 'gb2312']:
                try:
                    decoded_output = byte_data.decode(encoding)
                    cleaned_output = decoded_output.strip()
                    if cleaned_output:
                        self.log_box.append(cleaned_output)
                    return
                except UnicodeDecodeError:
                    continue
            
            # 如果都失败了，使用 ignore 模式
            decoded_output = byte_data.decode('utf-8', errors='ignore')
            cleaned_output = decoded_output.strip()
            if cleaned_output:
                self.log_box.append(cleaned_output)
                
        except Exception as e:
            # 最后的错误处理
            self.log_box.append(f"[日志处理错误: {str(e)}]")

    def handle_process_output(self, process):
        """处理进程输出"""
        output = process.readAllStandardOutput()
        self.log_output(output)

    def run_command(self, cmd, cwd=None):
        """运行命令，确保使用当前 Python 环境"""
        process = QProcess(self)
        
        # 设置环境变量确保正确编码
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONLEGACYWINDOWSFSENCODING", "1")
        process.setProcessEnvironment(env)
        
        # 连接输出信号
        process.readyReadStandardOutput.connect(lambda: self.handle_process_output(process))
        process.readyReadStandardError.connect(lambda: self.handle_process_output(process))
        
        # 设置工作目录
        if cwd:
            process.setWorkingDirectory(str(cwd))
        else:
            process.setWorkingDirectory(str(self.base_dir))
        
        # 在命令前添加 chcp 设置代码页（处理中文）
        full_cmd = f'chcp 65001 >nul & {cmd}'
        
        self.log_box.append(f"[调试] 执行命令: {full_cmd}")
        
        process.start("cmd.exe", ["/c", full_cmd])
        if not process.waitForStarted():
            self.log_box.append("❌ 命令启动失败")
        return process



    def upload_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "Audio Files (*.mp3 *.wav *.flac)")
        if not file_path:
            return

        original_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(original_name)[0]
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', name_without_ext)[:20] or 'audio'

        self.audio_name = safe_name
        self.audio_folder = self.base_dir / "Taudio" / self.audio_name
        
        # 如果文件夹已存在，先删除
        if self.audio_folder.exists():
            shutil.rmtree(self.audio_folder)
            self.log_box.append(f"🗑️ 已删除已存在的文件夹: {self.audio_folder}")
        
        os.makedirs(self.audio_folder, exist_ok=True)

        target_path = self.audio_folder / f"{self.audio_name}.mp3"
        shutil.copy(file_path, target_path)
        self.log_box.append(f"✅ 音频已上传并保存到: {target_path}")

    def generate_prompt(self):
        if not self.audio_name:
            QMessageBox.warning(self, "错误", "请先上传音频文件")
            return
            
        self.log_box.append("正在生成提示词...")
        
        # 清理可能已存在的提示词文件
        prompt_file = self.audio_folder / f"{self.audio_name}_prompt.txt"
        if prompt_file.exists():
            prompt_file.unlink()
            self.log_box.append(f"🗑️ 已删除已存在的提示词文件: {prompt_file}")
        
        # 不要给路径加引号，让 run_command 方法处理
        cmd = f'python generate_prompts_lyrics.py --data_dir {self.audio_folder}'
        process = self.run_command(cmd)
        process.finished.connect(lambda exit_code, exit_status: self.handle_script_completion(
            exit_code, exit_status, "提示词", prompt_file, self.display_prompt_content))

    def generate_lyrics(self):
        if not self.audio_name:
            QMessageBox.warning(self, "错误", "请先上传音频文件")
            return
            
        self.log_box.append("正在生成歌词...")
        
        # 清理可能已存在的歌词文件
        lyrics_file = self.audio_folder / f"{self.audio_name}_lyrics.txt"
        if lyrics_file.exists():
            lyrics_file.unlink()
            self.log_box.append(f"🗑️ 已删除已存在的歌词文件: {lyrics_file}")
        
        # 不要给路径加引号，让 run_command 方法处理
        cmd = f'python generate_prompts_lyrics.py --data_dir {self.audio_folder} --lyrics'
        process = self.run_command(cmd)
        process.finished.connect(lambda exit_code, exit_status: self.handle_script_completion(
            exit_code, exit_status, "歌词", lyrics_file, self.display_lyrics_content))

    def handle_script_completion(self, exit_code, exit_status, script_type, file_path, display_func):
        """处理脚本执行完成后的回调"""
        if exit_code == 0:  # 成功执行
            self.log_box.append(f"✅ {script_type}生成脚本执行完成")
            display_func()
        else:  # 执行失败
            self.log_box.append(f"❌ {script_type}生成脚本执行失败 (退出码: {exit_code})")
            self.log_box.append(f"💡 提示: 请检查是否安装了必要的依赖包")
            # 添加 Python 环境检查信息
            self.log_box.append(f"当前 Python: {sys.executable}")
            try:
                import torch
                self.log_box.append(f"PyTorch 已安装: {torch.__version__}")
            except ImportError:
                self.log_box.append("❌ PyTorch 未在当前环境中找到")

    def display_prompt_content(self):
        prompt_file = self.audio_folder / f"{self.audio_name}_prompt.txt"
        if prompt_file.exists():
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.prompt_display.setPlainText(content)
                self.log_box.append(f"✅ 提示词已加载到显示框")
            except Exception as e:
                self.log_box.append(f"❌ 读取提示词文件失败: {str(e)}")
        else:
            self.log_box.append(f"⚠️ 未找到提示词文件: {prompt_file} (可能生成失败)")

    def display_lyrics_content(self):
        lyrics_file = self.audio_folder / f"{self.audio_name}_lyrics.txt"
        if lyrics_file.exists():
            try:
                with open(lyrics_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.lyrics_display.setPlainText(content)
                self.log_box.append(f"✅ 歌词已加载到显示框")
            except Exception as e:
                self.log_box.append(f"❌ 读取歌词文件失败: {str(e)}")
        else:
            self.log_box.append(f"⚠️ 未找到歌词文件: {lyrics_file} (可能生成失败)")

    def create_dataset(self):
        if not self.audio_name:
            QMessageBox.warning(self, "错误", "请先上传音频文件")
            return
            
        self.log_box.append("正在创建数据集...")
        
        # 清理可能已存在的数据集文件夹
        output_name = self.audio_folder.parent / f"{self.audio_name}_filenames"
        if output_name.exists():
            shutil.rmtree(output_name)
            self.log_box.append(f"🗑️ 已删除已存在的数据集文件夹: {output_name}")
        
        cmd = f'python convert2hf_dataset_new.py --data_dir {self.audio_folder} --output_name {output_name}'
        process = self.run_command(cmd)
        process.finished.connect(lambda exit_code, exit_status: self.log_script_result(
            exit_code, exit_status, "数据集创建"))

    def preprocess_audio(self):
        if not self.audio_name:
            QMessageBox.warning(self, "错误", "请先上传音频文件")
            return
            
        self.log_box.append("正在进行音频预处理...")
        
        # 清理可能已存在的预处理文件夹
        output_dir = self.audio_folder.parent / f"{self.audio_name}_prep"
        if output_dir.exists():
            shutil.rmtree(output_dir)
            self.log_box.append(f"🗑️ 已删除已存在的预处理文件夹: {output_dir}")
        
        input_name = self.audio_folder.parent / f"{self.audio_name}_filenames"
        cmd = f'python preprocess_dataset_new.py --input_name {input_name} --output_dir {output_dir}'
        process = self.run_command(cmd)
        process.finished.connect(lambda exit_code, exit_status: self.log_script_result(
            exit_code, exit_status, "音频预处理"))

    def log_script_result(self, exit_code, exit_status, operation_name):
        """记录脚本执行结果"""
        if exit_code == 0:
            self.log_box.append(f"✅ {operation_name}完成")
        else:
            self.log_box.append(f"❌ {operation_name}失败 (退出码: {exit_code})")

    def start_training(self):
        if not self.update_params():
            return
        if not self.audio_name:
            QMessageBox.warning(self, "错误", "请先上传音频文件")
            return
            
        self.log_box.append("正在检查并清理 checkpoints 文件夹...")
        ckpt_dir = self.base_dir / "checkpoints"
        if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
            shutil.rmtree(ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.log_box.append("🗑️ 已清空 checkpoints 文件夹")
        else:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.log_box.append("正在启动训练...")
        dataset_path = self.audio_folder.parent / f"{self.audio_name}_prep"
        cmd = (
            f'python trainer_new.py --dataset_path {dataset_path} '
            f'--batch_size {self.params["batch_size"]} '
            f'--num_workers {self.params["num_workers"]} '
            f'--tag_dropout {self.params["tag_dropout"]} '
            f'--learning_rate {self.params["learning_rate"]} '
            f'--max_steps {self.params["max_steps"]} '
            f'--precision {self.params["precision"]} '  # 这里去掉了双引号
            f'--save_every_n_train_steps {self.params["save_every_n_train_steps"]}'
        )
        process = self.run_command(cmd)
        process.finished.connect(lambda exit_code, exit_status: self.log_script_result(
            exit_code, exit_status, "训练"))


    def start_tensorboard(self):
        self.log_box.append("正在启动 TensorBoard...")
        tb_logs_dir = self.base_dir / "tb_logs"
        if not tb_logs_dir.exists():
            tb_logs_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["tensorboard", "--logdir", str(tb_logs_dir)], shell=True)
        self.log_box.append("✅ TensorBoard 已启动，请访问 http://localhost:6006")

    def reset_all(self):
        reply = QMessageBox.question(
            self, "确认重置", "确定要重置所有内容吗？这将删除所有生成的文件和文件夹。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # 重置参数
            self.params = DEFAULT_PARAMS.copy()
            self.batch_size_edit.setText(str(self.params['batch_size']))
            self.num_workers_edit.setText(str(self.params['num_workers']))
            self.tag_dropout_edit.setText(str(self.params['tag_dropout']))
            self.learning_rate_edit.setText(str(self.params['learning_rate']))
            self.max_steps_edit.setText(str(self.params['max_steps']))
            self.precision_edit.setText(str(self.params['precision']))
            self.save_every_n_steps_edit.setText(str(self.params['save_every_n_train_steps']))
            
            # 清空显示内容
            self.prompt_display.clear()
            self.lyrics_display.clear()
            
            # 重置文件相关变量
            self.audio_name = ""
            self.audio_folder = ""
            
            # 删除生成的文件夹
            folders_to_delete = ["Taudio", "checkpoints", "tb_logs"]
            for folder_name in folders_to_delete:
                folder_path = self.base_dir / folder_name
                if folder_path.exists():
                    shutil.rmtree(folder_path)
                    self.log_box.append(f"🗑️ 已删除文件夹: {folder_name}")
            
            self.log_box.append("✅ 所有内容已重置")

    def check_python_environment(self):
        """检查 Python 环境"""
        self.log_box.append(f"Python 解释器: {sys.executable}")
        try:
            import torch
            self.log_box.append(f"✅ PyTorch 可用: {torch.__version__}")
        except ImportError:
            self.log_box.append("❌ PyTorch 未安装或不可用")
        
        try:
            import transformers
            self.log_box.append(f"✅ Transformers 可用: {transformers.__version__}")
        except ImportError:
            self.log_box.append("❌ Transformers 未安装或不可用")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ACEStepTrainerGUI()
    window.show()
    sys.exit(app.exec_())
