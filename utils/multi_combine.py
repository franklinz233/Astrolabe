import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_video(video_name, dir_paths, column_names, output_dir):
    """
    处理单个视频的拼接，并在每列顶部添加名称标签
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    paths = [Path(p) for p in dir_paths]
    valid_files = []
    for p in paths:
        file_path = p / video_name
        if file_path.exists():
            valid_files.append(str(file_path))
        else:
            return None  # 某个目录缺文件则跳过

    # 生成输出文件名
    output_file = Path(output_dir) / f"compare_{len(paths)}way_{video_name}"

    # 构建 ffmpeg 命令
    input_args = []
    for vf in valid_files:
        input_args.extend(['-i', vf])

    # 构建滤镜：缩放 + 添加文字标签
    filter_str = ""
    target_height = 720  # 视频高度
    text_height = 50      # 文字区域高度
    total_height = target_height + text_height
    
    for i in range(len(valid_files)):
        label = column_names[i] if i < len(column_names) else f"Video {i+1}"
        
        # 1. 缩放视频到目标高度
        # 2. 在顶部添加文字（使用 drawtext）
        # 3. 在视频顶部添加黑色背景条用于显示文字
        filter_str += (
            f"[{i}:v]scale=-2:{target_height}[scaled{i}];"
            f"[scaled{i}]pad=iw:ih+{text_height}:0:{text_height}:black[padded{i}];"
            f"[padded{i}]drawtext="
            f"text='{label}':"
            f"fontsize=32:"
            f"fontcolor=white:"
            f"x=(w-text_w)/2:"  # 水平居中
            f"y=10:"              # 距顶部10像素
            f"box=0:"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf[v{i}];"
        )
    
    # 横向拼接所有处理过的视频流
    concat_inputs = "".join([f"[v{i}]" for i in range(len(valid_files))])
    filter_str += f"{concat_inputs}hstack=inputs={len(valid_files)}[v]"

    cmd = [
        'ffmpeg', '-y',
        *input_args,
        '-filter_complex', filter_str,
        '-map', '[v]',
        '-c:v', 'libx264',
        '-crf', '26',
        '-pix_fmt', 'yuv420p',
        '-r', '24',
        str(output_file)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return video_name
    except subprocess.CalledProcessError as e:
        print(f"❌ 处理 {video_name} 时出错: {e.stderr.decode()}")
        return None

def multi_combine_videos_mp(dir_paths, column_names, output_dir, max_workers=8):
    """
    支持多进程加速的视频横向拼接，并在每列顶部显示名称。
    
    Args:
        dir_paths: 文件夹路径列表
        column_names: 每列对应的名称列表（长度应与 dir_paths 一致）
        output_dir: 输出目录
        max_workers: 最大进程数
    """
    if not dir_paths:
        print("错误：未提供文件夹路径。")
        return
    
    if len(column_names) != len(dir_paths):
        print(f"⚠️  警告：column_names 长度({len(column_names)}) 与 dir_paths 长度({len(dir_paths)}) 不一致")
        # 自动补齐或截断
        column_names = (column_names + [f"Video {i+1}" for i in range(len(dir_paths))])[:len(dir_paths)]

    os.makedirs(output_dir, exist_ok=True)
    paths = [Path(p) for p in dir_paths]
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    base_path = paths[0]
    video_names = [f.name for f in base_path.iterdir() if f.suffix.lower() in video_extensions]

    print(f"📊 检测到 {len(paths)} 个待对比目录")
    print(f"📹 基准目录中有 {len(video_names)} 个视频文件")
    print(f"🏷️  列名称: {column_names}")
    print(f"⚙️  使用 {max_workers} 个进程并行处理\n")

    count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_video, video_name, dir_paths, column_names, output_dir): video_name 
            for video_name in video_names
        }
        for future in as_completed(futures):
            video_name = futures[future]
            result = future.result()
            if result is not None:
                count += 1
                print(f"✅ [{count}/{len(video_names)}] 完成拼接: {video_name}")
            else:
                print(f"⏭️  跳过: {video_name} (文件缺失)")

    print(f"\n🎉 任务完成！共生成 {count} 个多路对比视频")
    print(f"📁 结果保存在: {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="横向拼接多个目录中的同名视频，并在顶部添加列名标签",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 两路对比，自动用目录名作为标签
  python multi_combine.py -d outputs/base/videos outputs/rl/videos -o outputs/compare

  # 指定列名
  python multi_combine.py -d outputs/base/videos outputs/rl/videos -n Base RL -o outputs/compare

  # 三路对比，8进程
  python multi_combine.py -d dir1 dir2 dir3 -n A B C -o outputs/compare -j 8
        """
    )
    parser.add_argument(
        "-d", "--dirs", nargs="+", required=True,
        metavar="DIR",
        help="待对比的视频目录列表（顺序对应列顺序）"
    )
    parser.add_argument(
        "-n", "--names", nargs="+",
        metavar="NAME",
        help="每列的标签名称（默认使用目录名）"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        metavar="OUTPUT_DIR",
        help="输出目录"
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=4,
        metavar="N",
        help="并行进程数（默认: 4）"
    )

    args = parser.parse_args()

    # 默认用目录名作为列名
    names = args.names if args.names else [Path(d).name for d in args.dirs]

    multi_combine_videos_mp(args.dirs, names, args.output, max_workers=args.workers)