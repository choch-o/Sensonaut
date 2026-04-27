import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime
import glob
import argparse

def load_scene_data(file_path):
    """加载场景数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_agent_data(file_path):
    """加载代理数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_timestamp(timestamp_str):
    """解析时间戳字符串为datetime对象"""
    try:
        # 处理不同的时间戳格式
        if '.' in timestamp_str:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        else:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None

def get_agent_trajectory(user001_dir):
    """获取代理的移动轨迹"""
    agent_files = glob.glob(os.path.join(user001_dir, "agent_data_*.json"))
    agent_files.sort()  # 按文件名排序
    
    trajectory = []
    for file_path in agent_files:
        try:
            data = load_agent_data(file_path)
            timestamp = parse_timestamp(data['timestamp'])
            if timestamp:
                trajectory.append({
                    'timestamp': timestamp,
                    'position': data['position'],
                    'rotation': data['rotation'],
                    'audio_level': data.get('audioLevel', 0)
                })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return trajectory

def create_rotation_matrix(rotation_degrees):
    """创建旋转矩阵"""
    rot_x = np.radians(rotation_degrees['x'])  # Pitch (俯仰角)
    rot_y = np.radians(rotation_degrees['y'])  # Yaw (偏航角)
    rot_z = np.radians(rotation_degrees['z'])  # Roll (翻滚角)
    
    # Rotation around X-axis (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rot_x), -np.sin(rot_x)],
        [0, np.sin(rot_x), np.cos(rot_x)]
    ])
    
    # Rotation around Y-axis (yaw)
    Ry = np.array([
        [np.cos(rot_y), 0, np.sin(rot_y)],
        [0, 1, 0],
        [-np.sin(rot_y), 0, np.cos(rot_y)]
    ])
    
    # Rotation around Z-axis (roll)
    Rz = np.array([
        [np.cos(rot_z), -np.sin(rot_z), 0],
        [np.sin(rot_z), np.cos(rot_z), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx

def draw_coordinate_system_2d(ax, position, rotation, axis_length=1.0, alpha=0.8):
    """在2D图上绘制坐标系统"""
    R = create_rotation_matrix(rotation)
    
    # Define local coordinate system vectors using rotation matrix
    x_vec = R @ np.array([1, 0, 0])
    y_vec = R @ np.array([0, 1, 0])
    z_vec = R @ np.array([0, 0, 1])
    
    # Draw local coordinate system using line segments (2D projection)
    C = np.array([position['x'], position['z']])
    
    # X-axis (red) - project to XZ plane
    x_proj = np.array([x_vec[0], x_vec[2]])
    x_proj = x_proj / np.linalg.norm(x_proj) * axis_length
    
    # Y-axis (green) - project to XZ plane
    y_proj = np.array([y_vec[0], y_vec[2]])
    y_proj = y_proj / np.linalg.norm(y_proj) * axis_length * 0.1
    
    # Z-axis (blue) - project to XZ plane
    z_proj = np.array([z_vec[0], z_vec[2]])
    z_proj = z_proj / np.linalg.norm(z_proj) * axis_length
    
    # Draw all three coordinate axes
    seg_x = np.stack([C, C + x_proj], axis=0)
    ax.plot(seg_x[:,0], seg_x[:,1], color='red', linewidth=2, alpha=alpha)
    
    seg_y = np.stack([C, C + y_proj], axis=0)
    ax.plot(seg_y[:,0], seg_y[:,1], color='green', linewidth=2, alpha=alpha)
    
    seg_z = np.stack([C, C + z_proj], axis=0)
    ax.plot(seg_z[:,0], seg_z[:,1], color='blue', linewidth=2, alpha=alpha)

def draw_coordinate_system_3d(ax, position, rotation, axis_length=1.0, alpha=0.8):
    """在3D图上绘制坐标系统"""
    R = create_rotation_matrix(rotation)
    
    # Define local coordinate system vectors using rotation matrix
    x_vec = R @ np.array([1, 0, 0])
    y_vec = R @ np.array([0, 1, 0])
    z_vec = R @ np.array([0, 0, 1])
    
    # Draw local coordinate system using line segments
    C = np.array([position['x'], 0, position['z']])
    
    # X-axis (red)
    seg_x = np.stack([C, C + x_vec * axis_length], axis=0)
    ax.plot(seg_x[:,0], seg_x[:,1], seg_x[:,2], color='red', linewidth=2, alpha=alpha)
    
    # Y-axis (green) - adjust length to match visual appearance
    y_axis_length = axis_length * 0.1
    seg_y = np.stack([C, C + y_vec * y_axis_length], axis=0)
    ax.plot(seg_y[:,0], seg_y[:,1], seg_y[:,2], color='green', linewidth=2, alpha=alpha)
    
    # Z-axis (blue)
    seg_z = np.stack([C, C + z_vec * axis_length], axis=0)
    ax.plot(seg_z[:,0], seg_z[:,1], seg_z[:,2], color='blue', linewidth=2, alpha=alpha)

def plot_vehicles_2d(ax, scene_data):
    """在2D图上绘制车辆"""
    for vehicle in scene_data['vehicles']:
        pos = vehicle['position']
        name = vehicle['name']
        
        # Choose color based on vehicle type
        if 'SoundSource' in name:
            color = 'red'
            edge_color = 'darkred'
        else:
            color = 'lightblue'
            edge_color = 'blue'
        
        # Draw car as rectangle with actual size (X: 2.5, Z: 4)
        car_width = 2.5  # X-axis size
        car_length = 4.0  # Z-axis size
        
        # Calculate rectangle corners
        x_left = pos['x'] - car_width/2
        x_right = pos['x'] + car_width/2
        z_bottom = pos['z'] - car_length/2
        z_top = pos['z'] + car_length/2
        
        # Create rectangle patch
        rect = patches.Rectangle((x_left, z_bottom), car_width, car_length, 
                               facecolor=color, edgecolor=edge_color, linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Add car label with background box to avoid overlap
        ax.text(pos['x'], pos['z'], name, fontsize=8, ha='center', va='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black'))

def plot_vehicles_3d(ax, scene_data):
    """在3D图上绘制车辆"""
    for vehicle in scene_data['vehicles']:
        pos = vehicle['position']
        name = vehicle['name']
        
        # Choose color based on vehicle type
        if 'SoundSource' in name:
            color = 'red'
            size = 0.8
        else:
            color = 'blue'
            size = 0.6
        
        # Plot vehicle position in 3D (Y fixed at 0 for XZ plane)
        ax.scatter3D(pos['x'], 0, pos['z'], c=color, s=100*size, alpha=0.8, 
                    edgecolors='black', linewidth=1, label=name if 'SoundSource' in name else None)
        ax.text(pos['x'], 0, pos['z'], name, fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black'))

def plot_agent_initial_position_2d(ax, scene_data):
    """在2D图上绘制代理初始位置"""
    agent = scene_data['agent']
    agent_pos = agent['position']
    ax.scatter(agent_pos['x'], agent_pos['z'], c='green', s=120, alpha=0.9, 
               edgecolors='black', linewidth=2, marker='o', label='Agent (Initial)')
    # 调整文字标签位置，避免被遮挡，放在代理右侧
    ax.text(agent_pos['x'] + 2.0, agent_pos['z'], 'Agent (Initial)', fontsize=10, 
            ha='left', va='center', weight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen'))

def plot_agent_initial_position_3d(ax, scene_data):
    """在3D图上绘制代理初始位置"""
    agent = scene_data['agent']
    agent_pos = agent['position']
    ax.scatter3D(agent_pos['x'], 0, agent_pos['z'], c='green', s=120, alpha=0.9, 
               edgecolors='black', linewidth=2, marker='o', label='Agent (Initial)')
    ax.text(agent_pos['x'], 0, agent_pos['z'], 'Agent (Initial)', fontsize=10, 
           ha='left', va='bottom', weight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))

def plot_trajectory_2d(ax, trajectory, show_direction_arrows=True):
    """在2D图上绘制轨迹"""
    if not trajectory:
        return
    
    # Extract position data - use X and Z for 2D plot
    x_positions = [point['position']['x'] for point in trajectory]
    z_positions = [point['position']['z'] for point in trajectory]
    
    # Plot 2D trajectory line (light blue)
    ax.plot(x_positions, z_positions, 'lightblue', linewidth=3, alpha=0.8, label='Agent 2D Path')
    
    # Plot trajectory points
    ax.scatter(x_positions, z_positions, c='green', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Mark start and end points
    ax.scatter(x_positions[0], z_positions[0], c='darkgreen', s=100, alpha=0.9, 
               edgecolors='black', linewidth=2, marker='s', label='Start')
    ax.scatter(x_positions[-1], z_positions[-1], c='red', s=100, alpha=0.9, 
               edgecolors='black', linewidth=2, marker='X', label='End')
    
    if show_direction_arrows:
        # Add direction arrows
        for i in range(0, len(trajectory), 10):  # Draw arrow every 10 points
            if i < len(trajectory) - 1:
                point = trajectory[i]
                next_point = trajectory[i + 1]
                
                # Calculate direction
                dx = next_point['position']['x'] - point['position']['x']
                dz = next_point['position']['z'] - point['position']['z']
                
                # Draw direction arrow
                ax.arrow(point['position']['x'], point['position']['z'], 
                         dx*0.3, dz*0.3, 
                         color='green', alpha=0.7, head_width=0.2, head_length=0.2, linewidth=2)

def plot_trajectory_3d(ax, trajectory, show_direction_arrows=True):
    """在3D图上绘制轨迹"""
    if not trajectory:
        return
    
    # Extract position data - use X and Z, keep Y fixed at 0 for XZ plane view
    x_positions = [point['position']['x'] for point in trajectory]
    z_positions = [point['position']['z'] for point in trajectory]
    y_positions = [0] * len(trajectory)  # Fixed Y value for XZ plane
    
    # Plot 3D trajectory line (light blue)
    ax.plot3D(x_positions, y_positions, z_positions, 'lightblue', linewidth=3, alpha=0.8, label='Agent 3D Path')
    
    # Plot trajectory points
    ax.scatter3D(x_positions, y_positions, z_positions, c='green', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Mark start and end points
    ax.scatter3D(x_positions[0], y_positions[0], z_positions[0], c='darkgreen', s=100, alpha=0.9, 
               edgecolors='black', linewidth=2, marker='s', label='Start')
    ax.scatter3D(x_positions[-1], y_positions[-1], z_positions[-1], c='red', s=100, alpha=0.9, 
               edgecolors='black', linewidth=2, marker='X', label='End')
    
    if show_direction_arrows:
        # Add direction arrows
        for i in range(0, len(trajectory), 10):  # Draw arrow every 10 points
            if i < len(trajectory) - 1:
                point = trajectory[i]
                next_point = trajectory[i + 1]
                
                # Calculate direction (only X and Z, Y is fixed)
                dx = next_point['position']['x'] - point['position']['x']
                dz = next_point['position']['z'] - point['position']['z']
                
                # Draw direction arrow
                ax.quiver(point['position']['x'], 0, point['position']['z'], 
                         dx*0.3, 0, dz*0.3, 
                         color='green', alpha=0.7, length=1.0, arrow_length_ratio=0.2, linewidth=2)

def set_2d_plot_properties(ax):
    """设置2D图属性"""
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('2D Vehicle and Agent Positions and Movement Trajectory (XZ Plane)')
    
    # Set axis ranges - fixed ranges as requested
    ax.set_xlim(25, 38)  # Fixed X-axis range
    ax.set_ylim(-14.5, 14.5)  # Fixed Z-axis range
    
    # Set equal aspect ratio to ensure grid squares are equal
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.legend()

def set_3d_plot_properties(ax):
    """设置3D图属性"""
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Vehicle and Agent Positions and Movement Trajectory (XZ Plane)')
    
    # Set axis ranges - fixed ranges as requested
    ax.set_xlim(25, 38)  # Fixed X-axis range
    ax.set_ylim(-0.5, 0.5)  # Small range around Y=0
    ax.set_zlim(-14.5, 14.5)  # Fixed Z-axis range
    
    # Set the view to show XZ plane (top-down view)
    ax.view_init(elev=0, azim=-90)
    
    # Add grid
    ax.grid(True, alpha=0.3)

def save_animation(animation, output_file, mode):
    """保存动画到文件"""
    try:
        # Use ffmpeg writer for MP4
        animation.save(output_file, writer='ffmpeg', fps=2, dpi=100)
        print(f"✅ Animation saved successfully to: {output_file}")
        print(f"📁 File location: {os.path.abspath(output_file)}")
        return True
    except Exception as e:
        print(f"❌ Error saving MP4: {e}")
        # Fallback to GIF
        gif_file = output_file.replace('.mp4', '.gif')
        print(f"🔄 Trying to save as GIF: {gif_file}")
        try:
            animation.save(gif_file, writer='pillow', fps=2)
            print(f"✅ Animation saved as GIF: {gif_file}")
            print(f"📁 File location: {os.path.abspath(gif_file)}")
            return True
        except Exception as e2:
            print(f"❌ Error saving GIF: {e2}")
            print("⚠️  Animation display only (not saved)")
            return False

def plot_2d_trajectory_with_rotation(trajectory, scene_data, ax):
    """Plot 2D version of agent movement trajectory in XZ plane with vehicle positions and rotation coordinate systems"""
    if not trajectory:
        return
    
    # Plot trajectory
    plot_trajectory_2d(ax, trajectory)
    
    # Plot agent rotation coordinate systems at key points
    for i in range(0, len(trajectory), 3):  # Draw rotation every 3 points to avoid clutter
        point = trajectory[i]
        draw_coordinate_system_2d(ax, point['position'], point['rotation'])
    
    # Plot vehicles and agent
    plot_vehicles_2d(ax, scene_data)
    plot_agent_initial_position_2d(ax, scene_data)
    
    # Set plot properties
    set_2d_plot_properties(ax)

def create_2d_animation(trajectory, scene_data, ax):
    """Create 2D animation showing agent movement process"""
    if not trajectory:
        return None
    
    # Set animation parameters
    frames = len(trajectory)
    
    def animate(frame):
        ax.clear()
        
        # Redraw scene with vehicles and agent initial position
        plot_vehicles_2d(ax, scene_data)
        plot_agent_initial_position_2d(ax, scene_data)
        
        # Plot trajectory up to current frame
        current_trajectory = trajectory[:frame+1]
        if current_trajectory:
            x_positions = [point['position']['x'] for point in current_trajectory]
            z_positions = [point['position']['z'] for point in current_trajectory]
            
            # Plot trajectory line
            ax.plot(x_positions, z_positions, 'lightblue', linewidth=3, alpha=0.8)
            ax.scatter(x_positions, z_positions, c='green', s=50, alpha=0.6, 
                       edgecolors='black', linewidth=0.5)
            
            # Mark current position
            current_point = current_trajectory[-1]
            ax.scatter(current_point['position']['x'], current_point['position']['z'], 
                      c='orange', s=150, alpha=0.9, edgecolors='black', linewidth=2, marker='o')
            
            # Show current rotation coordinate system at current position
            draw_coordinate_system_2d(ax, current_point['position'], current_point['rotation'], axis_length=0.8, alpha=0.9)
            
            # Show current time (top-left)
            timestamp_str = current_point['timestamp'].strftime("%H:%M:%S")
            ax.text(0.02, 0.98, f'Time: {timestamp_str}', transform=ax.transAxes, 
                     fontsize=12, verticalalignment='top', 
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))
            
            # Show current agent position info (top-right)
            ax.text(0.98, 0.98, f'Current Agent: ({current_point["position"]["x"]:.1f}, {current_point["position"]["z"]:.1f})', 
                     transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8, edgecolor='blue'))
        
        # Set chart properties
        set_2d_plot_properties(ax)
        ax.set_title('2D Agent Movement Animation (XZ Plane)')
    
    return FuncAnimation(ax.figure, animate, frames=frames, interval=500, repeat=True)

def plot_3d_trajectory_with_rotation(trajectory, scene_data, ax):
    """Plot 3D version of agent movement trajectory in XZ plane with vehicle positions and rotation coordinate systems"""
    if not trajectory:
        return
    
    # Plot trajectory
    plot_trajectory_3d(ax, trajectory)
    
    # Plot agent rotation coordinate systems at key points
    for i in range(0, len(trajectory), 3):  # Draw rotation every 3 points to avoid clutter
        point = trajectory[i]
        draw_coordinate_system_3d(ax, point['position'], point['rotation'])
    
    # Plot vehicles and agent
    plot_vehicles_3d(ax, scene_data)
    plot_agent_initial_position_3d(ax, scene_data)
    
    # Set plot properties
    set_3d_plot_properties(ax)

def create_3d_animation(trajectory, scene_data, ax):
    """Create 3D animation showing agent movement process"""
    if not trajectory:
        return None
    
    # Set animation parameters
    frames = len(trajectory)
    
    def animate(frame):
        ax.clear()
        
        # Redraw scene with vehicles and agent initial position
        plot_vehicles_3d(ax, scene_data)
        plot_agent_initial_position_3d(ax, scene_data)
        
        # Plot trajectory up to current frame
        current_trajectory = trajectory[:frame+1]
        if current_trajectory:
            x_positions = [point['position']['x'] for point in current_trajectory]
            z_positions = [point['position']['z'] for point in current_trajectory]
            y_positions = [0] * len(current_trajectory)
            
            # Plot trajectory line
            ax.plot3D(x_positions, y_positions, z_positions, 'lightblue', linewidth=3, alpha=0.8)
            ax.scatter3D(x_positions, y_positions, z_positions, c='green', s=50, alpha=0.6, 
                        edgecolors='black', linewidth=0.5)
            
            # Mark current position
            current_point = current_trajectory[-1]
            ax.scatter3D(current_point['position']['x'], 0, current_point['position']['z'], 
                         c='orange', s=150, alpha=0.9, edgecolors='black', linewidth=2, marker='o')
            
            # Show current rotation coordinate system at current position
            draw_coordinate_system_3d(ax, current_point['position'], current_point['rotation'], axis_length=0.8, alpha=0.9)
            
            # Show current time (top-left)
            timestamp_str = current_point['timestamp'].strftime("%H:%M:%S")
            ax.text2D(0.02, 0.98, f'Time: {timestamp_str}', transform=ax.transAxes, 
                     fontsize=12, verticalalignment='top', 
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))
            
            # Show current agent position info (top-right)
            ax.text2D(0.98, 0.98, f'Current Agent: ({current_point["position"]["x"]:.1f}, {current_point["position"]["z"]:.1f})', 
                     transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8, edgecolor='blue'))
        
        # Set chart properties
        set_3d_plot_properties(ax)
        ax.set_title('3D Agent Movement Animation (XZ Plane)')
    
    return FuncAnimation(ax.figure, animate, frames=frames, interval=500, repeat=True)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize agent movement trajectory')
    parser.add_argument('--mode', '-m', choices=['2d', '3d'], default='2d',
                       help='Visualization mode: 2d or 3d (default: 2d)')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save animation to file')
    parser.add_argument('--output', '-o', default='animation',
                       help='Output filename without extension (default: animation)')
    args = parser.parse_args()
    
    # Set font for display
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Data path
    user001_dir = "Assets/StreamingAssets/user001"
    
    # Load scene data
    scene_files = glob.glob(os.path.join(user001_dir, "scene_data_*.json"))
    if not scene_files:
        print("Scene data file not found")
        return
    
    scene_data = load_scene_data(scene_files[0])
    print(f"Loaded scene data: {scene_files[0]}")
    
    # Get agent trajectory
    trajectory = get_agent_trajectory(user001_dir)
    print(f"Found {len(trajectory)} trajectory points")
    
    if not trajectory:
        print("No agent trajectory data found")
        return
    
    if args.mode == '2d':
        # Create 2D chart with animation
        print("\nCreating 2D visualization...")
        fig = plt.figure(figsize=(16, 14))
        ax = plt.subplot(111)
        
        # Show initial static view first
        plot_2d_trajectory_with_rotation(trajectory, scene_data, ax)
        plt.title('2D Vehicle and Agent Positions and Movement Trajectory (XZ Plane)\nClick to start animation')
        plt.show()
        
        # Create 2D animation
        print("Creating 2D movement animation...")
        fig_anim = plt.figure(figsize=(16, 14))
        ax_anim = plt.subplot(111)
        animation = create_2d_animation(trajectory, scene_data, ax_anim)
        
        if animation:
            if args.save:
                # Save animation
                output_file = f"{args.output}_2d.mp4"
                print(f"Saving animation to {output_file}...")
                save_animation(animation, output_file, '2d')
            else:
                plt.show()
    
    else:  # 3D mode
        # Create 3D chart with animation
        print("\nCreating 3D visualization...")
        fig = plt.figure(figsize=(16, 14))
        ax = plt.subplot(111, projection='3d')
        
        # Show initial static view first
        plot_3d_trajectory_with_rotation(trajectory, scene_data, ax)
        plt.title('3D Vehicle and Agent Positions and Movement Trajectory (XZ Plane)\nClick to start animation')
        plt.show()
        
        # Create 3D animation
        print("Creating 3D movement animation...")
        fig_anim = plt.figure(figsize=(16, 14))
        ax_anim = plt.subplot(111, projection='3d')
        animation = create_3d_animation(trajectory, scene_data, ax_anim)
        
        if animation:
            if args.save:
                # Save animation
                output_file = f"{args.output}_3d.mp4"
                print(f"Saving animation to {output_file}...")
                save_animation(animation, output_file, '3d')
            else:
                plt.show()
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Number of vehicles in scene: {len(scene_data['vehicles'])}")
    print(f"Number of agent trajectory points: {len(trajectory)}")
    
    if trajectory:
        start_pos = trajectory[0]['position']
        end_pos = trajectory[-1]['position']
        total_distance = np.sqrt((end_pos['x'] - start_pos['x'])**2 + (end_pos['z'] - start_pos['z'])**2)
        print(f"Total movement distance: {total_distance:.2f} units")
        
        start_time = trajectory[0]['timestamp']
        end_time = trajectory[-1]['timestamp']
        duration = (end_time - start_time).total_seconds()
        print(f"Total movement time: {duration:.2f} seconds")
        print(f"Average speed: {total_distance/duration:.2f} units/second")

if __name__ == "__main__":
    main()

