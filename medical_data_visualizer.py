import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 read the dataframe
df  = pd.read_csv('medical_examination.csv')

# 2 adding an overweight column (weihght (kg) / height(m) **2)
BMI = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (BMI > 25).astype(int)
df.head()

# 3 normaize the cholesterol and the glucose column
for col in ['cholesterol','gluc']:
  df[col] =(df[col] >1 ).astype(int)

# OR
# for col in ['cholesterol', 'glucose']:
#     df.loc[df[col] == 1, col] = 0
#     df.loc[df[col] > 1, col] = 1

# OR

# # For 'cholesterol'
# df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
# df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1

# # For 'glucose'
# df.loc[df['gluc'] == 1, 'gluc'] = 0
# df.loc[df['gluc'] > 1, 'gluc'] = 1


# 4 draw the categorical plot
def draw_cat_plot():
    # 5: Melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6: Group manually
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7: Draw barplot with pre-counted data
    catplot = sns.catplot(
        data=df_cat,
        kind='bar',
        x='variable',
        y='total',
        hue='value',
        col='cardio'
    )

    # 8 get the figure
    fig = catplot.fig

    # 9 save
    fig.savefig('catplot.png')
    plt.close(fig)  # <-- Prevent duplicate display in Colab
    return fig



# 10 draw the heatmap
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <=df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
]


    # 12
    corr = df.corr(numeric_only=True)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14 setup matplot lib figure
    fig, ax = plt.subplots(figsize=(12, 10))


    # 15 draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', ax=ax,cbar_kws={"shrink":0.5})


    # 16 save and return
    fig.savefig('heatmap.png')
    plt.close(fig)  # <-- Prevent duplicate display in Colab
    return fig
