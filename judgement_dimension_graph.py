import numpy as np
import matplotlib.pyplot as plt

barWidth = 0.15
fig = plt.subplots(figsize =(12, 8))

dim1 = [69.2, 49.1]
dim2 = [71.2, 70.3]
dim3 = [72.9, 77.2]
dim4 = [86.7, 82.2]
dim5 = [90.4, 87.7]

br1 = np.arange(len(dim1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]

bar1 = plt.bar(br1, dim1, color ='r', width = barWidth, 
        edgecolor ='grey', label ='1-dim') 
bar2 = plt.bar(br2, dim2, color ='g', width = barWidth, 
        edgecolor ='grey', label ='2-dim') 
bar3 = plt.bar(br3, dim3, color ='b', width = barWidth, 
        edgecolor ='grey', label ='3-dim')
bar4 = plt.bar(br4, dim4, color ='violet', width = barWidth, 
        edgecolor ='grey', label ='4-dim')
bar5 = plt.bar(br5, dim5, color ='c', width = barWidth, 
        edgecolor ='grey', label ='5-dim') 

plt.title('Judgement Dimension', fontweight='bold', fontsize=20)

plt.xlabel('Dataset', fontweight ='bold', fontsize = 15) 
plt.ylabel('F1 (%)', fontweight ='bold', fontsize = 15) 
plt.xticks([r + 2 * barWidth for r in range(len(dim1))], 
        ['helpsteer2', 'neurips'])

plt.bar_label(bar1, padding=3)
plt.bar_label(bar2, padding=3)
plt.bar_label(bar3, padding=3)
plt.bar_label(bar4, padding=3)
plt.bar_label(bar5, padding=3)

plt.legend()
plt.show()
