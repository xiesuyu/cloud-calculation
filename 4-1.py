import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 加载数据
crime_data = pd.read_csv('crimeRatesByState2005.csv')
birth_data = pd.read_csv('birthrate.csv')

# 1. 绘制散点图及拟合曲线
plt.figure(figsize=(10, 6))
sns.scatterplot(x='murder', y='burglary', data=crime_data, color='green', marker='^')
# 过滤掉离散数据点（这里简化处理，假设我们移除NaN值）
crime_data_filtered = crime_data.dropna(subset=['murder', 'burglary'])
# 添加拟合曲线
slope, intercept, r_value, p_value, std_err = stats.linregress(crime_data_filtered['murder'],crime_data_filtered['burglary'])
fit_line = slope * crime_data_filtered['murder'] + intercept
plt.plot(crime_data_filtered['murder'], fit_line, color='green',label=f'y = {slope:.2f}x + {intercept:.2f}, R^2={r_value ** 2:.2f}')
plt.xlabel('Murder Rate')
plt.ylabel('Burglary Rate')
plt.title('Murder vs Burglary Scatter Plot with Fitted Line')
plt.legend()
plt.show()

# 2. 绘制散点矩阵图
sns.pairplot(crime_data, vars=['murder', 'burglary', 'assault', 'robbery', 'rape', 'larceny', 'auto_theft'], kind='reg')
plt.show()

# 3. 绘制气泡图
for crime_type in ['assault', 'robbery', 'rape', 'larceny', 'auto_theft']:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='murder', y=crime_type, data=crime_data, hue='population', size='population', sizes=(20, 200),palette='random', edgecolor='black', linewidth=0.6, alpha=0.6)
    plt.xscale('log')  # 如果需要的话，可以使用对数尺度
    plt.yscale('log')  # 如果需要的话，可以使用对数尺度
    plt.xlabel('Murder Rate')
    plt.ylabel(f'{crime_type.capitalize()} Rate')
    plt.title(f'Murder vs {crime_type.capitalize()} Bubble Plot')
    plt.show()


# 4. 绘制茎叶图
# 注意：matplotlib没有内置的茎叶图函数，但我们可以使用seaborn的factorplot或自定义函数来绘制。这里使用自定义函数作为示例。
def draw_stem_and_leaf_plot(data, value_column):
    # 这里简化处理，仅展示如何绘制一个基本的茎叶图，实际实现可能需要更复杂的逻辑
    values = data[value_column].sort_values()
    stem_width = 10  # 茎的宽度
    stem_values = values // stem_width
    leaf_values = values % stem_width
    fig, ax = plt.subplots()
    y_pos = range(len(stem_values))
    ax.barh(y_pos, stem_values, color='gray', alpha=0.6)
    for i, (stem, leaf) in enumerate(zip(stem_values, leaf_values)):
        ax.text(stem + (leaf / float(stem_width)) * 0.3, i, str(leaf), va='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stem_values)
    ax.invert_yaxis()  # 翻转y轴，使得较小的值在顶部
    ax.set_xlabel('Birth Rate')
    ax.set_title(f'Stem and Leaf Plot of {value_column.capitalize()}')
    plt.show()

    #选择一个列来绘制茎叶图，例如birthrate
    draw_stem_and_leaf_plot(birth_data, 'birthrate')

    #绘制直方图和密度图
    plt.figure(figsize=(14, 7))

    #直方图
    plt.subplot(1, 2, 1)
    sns.histplot(birth_data['birthrate'], kde=True, bins=30, color='lightblue', fill=True)
    plt.xlabel('Birth Rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of Birth Rate')

    #密度图
    plt.subplot(1, 2, 2)
    sns.kdeplot(birth_data['birthrate'], bw_adjust=0.5, color='darkblue', label='KDE')
    plt.xlabel('Birth Rate')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimate of Birth Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()
