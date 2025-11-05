import os
import sys
import subprocess
import shutil
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QTextEdit,
    QVBoxLayout, QHBoxLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
    QFormLayout, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
import webbrowser
import re


class WorkerThread(QThread):
    output = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, command, cwd=None):
        super().__init__()
        self.command = command
        self.cwd = cwd

    def run(self):
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True,
                cwd=self.cwd
            )
            for line in process.stdout:
                self.output.emit(line.strip())
            process.wait()
            if process.returncode == 0:
                self.finished.emit()
            else:
                self.error.emit("命令执行失败")
        except Exception as e:
            self.error.emit(str(e))


class ACEStepGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ACE-Step 模型训练 GUI")
        self.resize(900, 700)

        # 基础目录
        self.base_audio_dir = r"D:\AIJOB\ACE-Step-T\ACE-Step\Taudio"
        self.current_audio_name = ""  # 不带扩展名的文件名
        self.current_audio_full_path = ""  # 完整路径
        
        self.step_status = [False] * 6

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 上传音频文件
        self.audio_file_label = QLabel("训练音频文件：未选择")
        self.upload_audio_btn = QPushButton("上传音频文件")
        self.upload_audio_btn.clicked.connect(self.upload_audio_file)

        layout.addWidget(self.audio_file_label)
        layout.addWidget(self.upload_audio_btn)

        # 按钮区域
        buttons_layout = QHBoxLayout()
        self.step_buttons = []

        step_names = [
            "1. 生成提示词",
            "2. 生成歌词",
            "3. 创建文件名数据集",
            "4. 音频预处理",
            "5. 开始训练"
        ]
        for i, name in enumerate(step_names):
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, idx=i: self.run_step(idx))
            btn.setEnabled(False)
            buttons_layout.addWidget(btn)
            self.step_buttons.append(btn)

        layout.addLayout(buttons_layout)

        # TensorBoard & Reset
        extra_layout = QHBoxLayout()
        self.tensorboard_btn = QPushButton("启动 TensorBoard")
        self.tensorboard_btn.clicked.connect(self.start_tensorboard)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_all)
        extra_layout.addWidget(self.tensorboard_btn)
        extra_layout.addWidget(self.reset_btn)
        layout.addLayout(extra_layout)

        # 训练参数设置
        param_group = QGroupBox("训练参数设置")
        param_layout = QFormLayout()

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setValue(1)
        param_layout.addRow(QLabel("Batch Size"), self.batch_size_spin)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setValue(0)
        param_layout.addRow(QLabel("Num Workers"), self.num_workers_spin)

        self.tag_dropout_spin = QDoubleSpinBox()
        self.tag_dropout_spin.setRange(0.0, 1.0)
        self.tag_dropout_spin.setSingleStep(0.1)
        self.tag_dropout_spin.setValue(0.5)
        param_layout.addRow(QLabel("Tag Dropout"), self.tag_dropout_spin)

        self.learning_rate_edit = QLineEdit("1e-4")
        param_layout.addRow(QLabel("Learning Rate"), self.learning_rate_edit)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setMaximum(100000)
        self.max_steps_spin.setValue(2000)
        param_layout.addRow(QLabel("Max Steps"), self.max_steps_spin)

        self.precision_edit = QLineEdit("bf16-mixed")
        param_layout.addRow(QLabel("Precision"), self.precision_edit)

        self.save_steps_spin = QSpinBox()
        self.save_steps_spin.setValue(100)
        param_layout.addRow(QLabel("Save Every N Steps"), self.save_steps_spin)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 日志区域
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def log(self, text):
        self.log_area.append(text)

    def chinese_to_pinyin_initials(self, text):
        """中文转拼音首字母"""
        pinyin_map = {
            '张': 'z', '雨': 'y', '生': 's', '一': 'y', '天': 't', 
            '到': 'd', '晚': 'w', '游': 'y', '泳': 'y', '的': 'd', '鱼': 'y',
            '二': 'e', '三': 's', '四': 's', '五': 'w', '六': 'l', '七': 'q', '八': 'b', '九': 'j', '十': 's',
            '是': 's', '我': 'w', '你': 'n', '他': 't', '她': 't', '它': 't', '们': 'm',
            '好': 'h', '很': 'h', '了': 'l', '么': 'm', '呢': 'n', '吧': 'b', '啊': 'a',
            '爱': 'a', '情': 'q', '心': 'x', '梦': 'm', '想': 'x', '希': 'x', '望': 'w',
            '快': 'k', '乐': 'l', '悲': 'b', '伤': 's', '高': 'g', '兴': 'x',
            '美': 'm', '丽': 'l', '漂': 'p', '亮': 'l', '聪': 'c', '明': 'm',
            '大': 'd', '小': 'x', '中': 'z', '上': 's', '下': 'x', '左': 'z', '右': 'y',
            '前': 'q', '后': 'h', '里': 'l', '外': 'w', '内': 'n', '东': 'd', '西': 'x', '南': 'n', '北': 'b'
        }
        
        result = ""
        for char in text:
            if char in pinyin_map:
                result += pinyin_map[char]
            elif char.isalnum():
                result += char.lower()
            # 跳过空格和特殊符号
        
        return result

    def get_safe_filename(self, original_name):
        """获取安全的文件名（10字符以内，完全英文数字）"""
        # 去掉扩展名
        name_without_ext = os.path.splitext(original_name)[0]
        
        # 移除艺术家信息（通常在 - 前面）
        if ' - ' in name_without_ext:
            name_without_ext = name_without_ext.split(' - ')[-1]
        
        # 如果包含中文，转换为拼音首字母
        if re.search(r'[\u4e00-\u9fff]', name_without_ext):
            safe_name = self.chinese_to_pinyin_initials(name_without_ext)
        else:
            # 英文名也做清理处理
            safe_name = name_without_ext.lower()
        
        # 只保留字母和数字
        safe_name = re.sub(r'[^a-z0-9]', '', safe_name)
        
        # 限制长度为10个字符
        if len(safe_name) > 10:
            safe_name = safe_name[:10]
        elif len(safe_name) == 0:
            safe_name = "audio"
            
        return safe_name

    def upload_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择音频文件", 
            "", 
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.aac)"
        )
        
        if file_path:
            original_filename = os.path.basename(file_path)
            safe_name = self.get_safe_filename(original_filename)
            
            # 创建目标目录
            target_dir = os.path.join(self.base_audio_dir, safe_name)
            
            # 检查目录是否存在
            if os.path.exists(target_dir):
                reply = QMessageBox.question(
                    self,
                    "确认",
                    f"目录 {target_dir} 已存在，是否清空并继续？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                else:
                    shutil.rmtree(target_dir)
            
            os.makedirs(target_dir, exist_ok=True)
            
            # 目标文件路径
            file_ext = os.path.splitext(original_filename)[1]
            target_file_path = os.path.join(target_dir, safe_name + file_ext)
            
            # 复制文件
            try:
                shutil.copy2(file_path, target_file_path)
                self.current_audio_name = safe_name
                self.current_audio_full_path = target_file_path
                
                self.audio_file_label.setText(f"训练音频文件：{original_filename} -> {safe_name}{file_ext}")
                self.log(f"音频文件已上传到: {target_file_path}")
                
                # 启用后续步骤按钮
                self.update_step_buttons()
                
            except Exception as e:
                self.log(f"文件上传失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"文件上传失败: {str(e)}")

    def update_step_buttons(self):
        for i in range(5):
            self.step_buttons[i].setEnabled(True)

    def run_step(self, step_index):
        if step_index > 0 and not self.step_status[step_index - 1]:
            self.log("请按顺序执行步骤！")
            return

        if not self.current_audio_name:
            self.log("请先上传音频文件！")
            return

        self.step_buttons[step_index].setEnabled(False)
        self.log(f"开始执行步骤 {step_index + 1}...")

        cmd = ""
        cwd = os.path.dirname(os.path.abspath(__file__))
        audio_dir_with_name = os.path.join(self.base_audio_dir, self.current_audio_name)

        if step_index == 0:
            cmd = f'python generate_prompts_lyrics.py --data_dir "{audio_dir_with_name}"'
        elif step_index == 1:
            cmd = f'python generate_prompts_lyrics.py --data_dir "{audio_dir_with_name}" --lyrics'
        elif step_index == 2:
            output_name = audio_dir_with_name + "_filenames"
            cmd = f'python convert2hf_dataset_new.py --data_dir "{audio_dir_with_name}" --output_name "{output_name}"'
        elif step_index == 3:
            input_name = audio_dir_with_name + "_filenames"
            output_dir = audio_dir_with_name + "_prep"
            cmd = f'python preprocess_dataset_new.py --input_name "{input_name}" --output_dir "{output_dir}"'
        elif step_index == 4:
            dataset_path = audio_dir_with_name + "_prep"
            cmd = (
                f'python trainer_new.py '
                f'--dataset_path "{dataset_path}" '
                f'--batch_size {self.batch_size_spin.value()} '
                f'--num_workers {self.num_workers_spin.value()} '
                f'--tag_dropout {self.tag_dropout_spin.value()} '
                f'--learning_rate {self.learning_rate_edit.text()} '
                f'--max_steps {self.max_steps_spin.value()} '
                f'--precision "{self.precision_edit.text()}" '
                f'--save_every_n_train_steps {self.save_steps_spin.value()}'
            )

        self.worker_thread = WorkerThread(cmd, cwd)
        self.worker_thread.output.connect(self.log)
        self.worker_thread.error.connect(self.log)
        self.worker_thread.finished.connect(lambda: self.on_step_finished(step_index))
        self.worker_thread.start()

    def on_step_finished(self, step_index):
        self.step_status[step_index] = True
        self.log(f"步骤 {step_index + 1} 完成。")

        # 如果是训练完成（第5步）
        if step_index == 4:
            self.process_lora_alpha()

        if step_index < 4:
            self.step_buttons[step_index + 1].setEnabled(True)


    def start_tensorboard(self):
        try:
            subprocess.Popen(["tensorboard", "--logdir", "tb_logs"])
            webbrowser.open("http://localhost:6006")
        except Exception as e:
            self.log(f"启动 TensorBoard 失败: {e}")

    def reset_all(self):
        self.current_audio_name = ""
        self.current_audio_full_path = ""
        self.step_status = [False] * 6
        self.audio_file_label.setText("训练音频文件：未选择")
        for btn in self.step_buttons:
            btn.setEnabled(False)
        self.log_area.clear()
        self.log("已重置所有设置。")

    def process_lora_alpha(self):
        """训练完成后自动处理 LoRA 权重"""
        try:
            # 获取训练输出路径（假设是与 prep 目录同级的 checkpoints）
            dataset_path = os.path.join(self.base_audio_dir, self.current_audio_name + "_prep")
            checkpoint_dir = os.path.join(os.path.dirname(dataset_path), "checkpoints")

            if not os.path.exists(checkpoint_dir):
                self.log("未找到 checkpoints 目录，跳过 LoRA alpha 处理。")
                return

            # 查找最新的 step 文件夹
            step_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch=") and os.path.isdir(os.path.join(checkpoint_dir, d))]
            if not step_dirs:
                self.log("未找到任何 epoch-step 目录，跳过 LoRA alpha 处理。")
                return

            latest_dir = sorted(step_dirs, key=lambda x: int(x.split("step=")[-1].split("_")[0]))[-1]
            lora_dir = os.path.join(checkpoint_dir, latest_dir)
            input_lora_path = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")

            if not os.path.exists(input_lora_path):
                self.log(f"未找到 LoRA 文件: {input_lora_path}")
                return

            # 输出文件路径
            output_lora_path = os.path.join(lora_dir, "pytorch_lora_weights_with_alpha.safetensors")

            # 配置文件路径（假设 config 文件在项目根目录）
            lora_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "lora_config_transformer_only.json")

            if not os.path.exists(lora_config_path):
                self.log(f"未找到 LoRA 配置文件: {lora_config_path}")
                return

            # 构建命令
            cmd = (
                f'python add_alpha_in_lora.py '
                f'--input_name "{input_lora_path}" '
                f'--output_name "{output_lora_path}" '
                f'--lora_config_path "{lora_config_path}"'
            )

            self.log("开始处理 LoRA 权重，写入 alpha 信息...")
            self.log(f"执行命令: {cmd}")

            result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
            if result.returncode == 0:
                self.log("✅ LoRA alpha 处理成功完成！")
                self.log(f"新 LoRA 文件保存为: {output_lora_path}")
            else:
                self.log("❌ LoRA alpha 处理失败:")
                self.log(result.stderr)

        except Exception as e:
            self.log(f"处理 LoRA alpha 时发生异常: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ACEStepGUI()
    window.show()
    sys.exit(app.exec())
