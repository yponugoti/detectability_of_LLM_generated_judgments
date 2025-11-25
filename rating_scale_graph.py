import numpy as np
import matplotlib.pyplot as plt

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

small_scale = [87.4, 52.4]
full_scale = [90.4, 63.1]

br1 = np.arange(len(small_scale))
br2 = [x + barWidth for x in br1]

bar1 = plt.bar(br1, small_scale, color ='r', width = barWidth, 
        edgecolor ='grey', label ='w/ small scale') 
bar2 = plt.bar(br2, full_scale, color ='g', width = barWidth, 
        edgecolor ='grey', label ='full') 

plt.title('Rating Scale', fontweight='bold', fontsize=20)

plt.xlabel('Dataset', fontweight ='bold', fontsize = 15) 
plt.ylabel('F1 (%)', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth/2 for r in range(len(small_scale))], 
        ['helpsteer2', 'helpsteer3'])

plt.bar_label(bar1, padding=3)
plt.bar_label(bar2, padding=3)

plt.legend()
plt.show()
