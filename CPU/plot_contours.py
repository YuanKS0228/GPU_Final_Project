import numpy as np
import matplotlib.pyplot as plt

def main():
    x_coords = []
    y_coords = []
    density = []
    
    print("正在讀取 results.txt...")
    try:
        with open('results.txt', 'r') as f:
            for line in f:
                parts = line.replace('\t', '').strip().split(',')
                if len(parts) == 3:
                    x_coords.append(float(parts[0]))
                    y_coords.append(float(parts[1]))
                    density.append(float(parts[2]))
    except FileNotFoundError:
        print("找不到 results.txt！請確認求解器已成功執行並產生檔案。")
        return

    # 將原始資料轉換為 numpy array
    x_arr = np.array(x_coords)
    y_arr = np.array(y_coords)
    d_arr = np.array(density)

    # 找出網格點的獨立坐標，用以估算正確的 DX 與 DY
    unique_x = np.unique(x_arr)
    unique_y = np.unique(y_arr)
    dx = 3.0 / (len(unique_x)) if len(unique_x) > 0 else 0.01
    dy = 1.0 / (len(unique_y)) if len(unique_y) > 0 else 0.01

    # ---- 💡 解決白邊核心：手動補充上下左右的物理邊界點 ----
    extended_x = list(x_coords)
    extended_y = list(y_coords)
    extended_d = list(density)

    # 找出頂部和底部邊緣的點，將其延伸到 Y = 1.0 和 Y = 0.0
    for x, y, d in zip(x_coords, y_coords, density):
        if y == unique_y.max(): # 靠近頂部的點
            extended_x.append(x)
            extended_y.append(1.0) # 補齊到天花板
            extended_d.append(d)
        if y == unique_y.min(): # 靠近底部的點
            extended_x.append(x)
            extended_y.append(0.0) # 補齊到地板
            extended_d.append(d)
            
    # 找出左側和右側邊緣的點，將其延伸到 X = 0.0 和 X = 3.0
    for x, y, d in zip(x_coords, y_coords, density):
        if x == unique_x.min():
            extended_x.append(0.0)
            extended_y.append(y)
            extended_d.append(d)
        if x == unique_x.max():
            extended_x.append(3.0)
            extended_y.append(y)
            extended_d.append(d)

    # 2. 建立畫布
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    
    # 將背景顏色安全地設為基底深藍色
    ax.set_facecolor(plt.cm.jet(0.0))
    
    # 3. 繪製密度等高線圖（使用延伸後的數據，使色彩能完美填滿物理邊界）
    contour = plt.tricontourf(extended_x, extended_y, extended_d, levels=500, cmap='jet')
    plt.colorbar(contour, label='Density')

    # 4. 繪製並填滿階梯幾何形狀 (Step Geometry)
    # 完全重合你 main.cu 第 74/97 行的實體判定邊界
    step_start_x = 0.6 + dx
    step_height_y = 0.2 + dy
    
    step_x = [step_start_x, step_start_x, 3.0, 3.0]
    step_y = [0.0,          step_height_y, step_height_y, 0.0]
    
    # 填滿灰色階梯
    plt.fill(step_x, step_y, color='gray', zorder=20) 
    
    # 畫出階梯外框粗黑線
    plt.plot([step_start_x, step_start_x, 3.0], [0.0, step_height_y, step_height_y], 
             color='black', linewidth=2.5, zorder=25)

    # 5. 設定圖表標籤與嚴格物理邊界
    plt.title('Density Contours (Mach 3 Flow over a Forward-Facing Step)')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    
    # 嚴格且完美地切在 L=3.0, H=1.0 的物理視窗上
    plt.xlim(0.0, 3.0)
    plt.ylim(0.0, 1.0)
    ax.set_aspect('equal')

    # 6. 儲存圖片
    output_filename = 'density.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"等高線圖已完美儲存為 {output_filename}")

if __name__ == "__main__":
    main()
