import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# data_path = 'data/raw/HDB5V2.3.xlsx'
data_path = 'data/raw/HDB5.2.3.csv'

def preprocess(path):
    # df_raw = pd.read_excel(path)
    df_raw = pd.read_csv(path)
    std5 = df_raw[df_raw["SELDB5"] == True].copy()

    # Absolute value except for triangularity parameters
    numeric_cols = std5.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols_no_delta = [col for col in numeric_cols if col not in ['DELTA', 'DELTAU', 'DELTAL']]
    std5[numeric_cols_no_delta] = std5[numeric_cols_no_delta].abs()
    std5['STD3'] = std5['STD3'].fillna(0)
    
    # Drop NaN
    std5['NEL'] = std5['NEL'].replace('', np.nan)
    std5 = std5.dropna(subset=["NEL"])
    print("After drop NEL")
    print(len(std5))
    std5 = std5.dropna(subset=['TAUTH'])
    print("After drop NaN TAUTH")
    print(len(std5))
    std5 = std5.dropna(subset=['DELTA'])
    print("After drop NaN DELTA")
    print(len(std5))

    # N19 and N20
    std5['NEL'] = std5['NEL'].astype('float64')
    std5['NEL'] = std5['NEL'] * 1e-19 # [m-3] -> [1e-19 m-3]
    std5['N20'] = std5['NEL'] * 0.1 # [m-3] -> [1e-20 m-3]
    
    # IP: A -> MA
    std5['IP'] = std5['IP'] * 1e-6 # [A] -> [MA]
    
    # POWER: W -> MW
    std5['POHM'] = std5['POHM'] * 1e-6 # [W] -> [MW]
    std5['PNBI'] = std5['PNBI'] * 1e-6 # [W] -> [MW]
    std5['PECRH'] = std5['PECRH'] * 1e-6 # [W] -> [MW]
    std5['PICRH'] = std5['PICRH'] * 1e-6 # [W] -> [MW]
    std5['PLTH'] = std5['PLTH'] * 1e-6 # [W] -> [MW]
    std5['PRAD'] = std5['PRAD'] * 1e-6 
    std5['PALPHA'] = std5['PALPHA'].fillna(0.0)
    std5['PALPHA'] = std5['PALPHA'] * 1e-6
    
    # EPSILON
    std5['EPSILON'] = std5['AMIN'] / std5['RGEO']
    
    # IPB98Y2 scaling law
    std5['TAUC92'] = std5['TAUC92'].fillna(1)
    std5['TAU98Y2'] = 0.0562 * \
                    (std5['IP'] ** 0.93) * \
                    (std5['BT'] ** 0.15) * \
                    (std5['NEL'] ** 0.41) * \
                    (std5['PLTH'] ** -0.69) * \
                    (std5['RGEO'] ** 1.97) * \
                    (std5['KAREA'] ** 0.78) * \
                    (std5['EPSILON'] ** 0.58) * \
                    (std5['MEFF'] ** 0.19)
    std5['HIPB98Y2'] = std5['HIPB98Y2'].fillna(std5['TAUTH'] * std5['TAUC92'] / std5['TAU98Y2'])
    
    # ITPA20 scaling law
    std5['TAU20'] = 0.053 * \
                    (std5['IP'] ** 0.98) * \
                    (std5['BT'] ** 0.22) * \
                    (std5['PLTH'] ** -0.67) * \
                    (std5['NEL'] ** 0.24) * \
                    (std5['MEFF'] ** 0.20) * \
                    (std5['RGEO'] ** 1.71) * \
                    (std5['KAREA'] ** 0.80) * \
                    (std5['EPSILON'] ** 0.35) * \
                    ((std5['DELTA'] + 1) ** 0.36)
    std5['H20'] = std5['TAUTH'] / std5['TAU20']
    
    # Martin Scaling
    std5['PLH'] = 2.15 * \
                  (std5['N20'] ** 0.782) * \
                  (std5['BT'] ** 0.772) * \
                  (std5['AMIN'] ** 0.975) * \
                  (std5['RGEO']) * \
                  (2/std5['MEFF'])
    # std5['PLH'] = 0.0488 * \
    #               (std5['N20'] ** 0.717) * \
    #               (std5['BT'] ** 0.803) * \
    #               (std5['SURFFORM'] ** 0.941) * \
    #               (2/std5['MEFF'])
                  
    # # L-H threshold for ST 
    # std5['PLH-MAST'] = 11.35 * (std5['N20'] ** 1.19) * (2/std5['MEFF'])
    # std5['PLH-NSTX'] = 5 * std5['PMARTIN']

    # # Define PLH based on TOK
    # conditions = [
    #     (std5['TOK'] == 'MAST'),
    #     (std5['TOK'].isin(['NSTX', 'START']))
    # ]
    # choices = [
    #     std5['PLH-MAST'],
    #     std5['PLH-NSTX']
    # ]
    # std5['PLH'] = np.select(conditions, choices, default=std5['PMARTIN'])

    new_columns = {
        # impurity
        'HIGHZ': np.where((std5['DIVMAT'] == "MO") | ((std5['DIVMAT'] == "W") & (std5['TOK'] == "JET")) | ((std5['LIMMAT'] == "W") & (std5["TOK"] == "AUG")), 1, 0),
        'COAT': np.where(std5['EVAP'] != 'NONE', 1, 0),

        # Magnetics
        'qCYL': 5 * (std5['BT'] / std5['IP']) * (std5['AMIN'] ** 2 / std5['RGEO']) * std5['KAREA'],
        'IPN': (std5['IP']) / (std5['BT'] * std5['AMIN']),
        
        # Density
        'GWF': std5['N20'] / (std5['IP'] / (np.pi * std5['AMIN'] ** 2)),

        # Input power
        'PFIN': (std5['PNBI'] + std5['PALPHA']) / std5['PLH'],
        'PRFN': (std5['PECRH'] + std5['PICRH']) / std5['PLH'],

        # Group
        'GROUP': np.where(std5['TOK'].isin(['NSTX', 'START', 'MAST']), 'ST',
                 np.where(std5['TOK'] == 'CMOD', 'COMPACT',
                 np.where(std5['TOK'].isin(['ASDEX', 'PDX', 'PBXM', 'T10', 'TDEV', 'TEXTOR', 'TFTR', 'TUMAN3M']), 'LEGACY',
                 np.where(std5['TOK'].isin(['AUG', 'COMPASS', 'D3D', 'JET', 'JFT2M', 'JT60U', 'TCV']), 'MODERN', 'OTHER'))))
    }
    
    # 기존 DataFrame과 새 열들을 한 번에 결합
    std5 = pd.concat([std5, pd.DataFrame(new_columns, index=std5.index)], axis=1)

    # JET & HIGHZ == 1 → JETILW, AUG & HIGHZ == 1 → AUGW
    std5.loc[(std5['TOK'] == 'JET') & (std5['HIGHZ'] == 1), 'TOK'] = 'JETILW'
    std5.loc[(std5['TOK'] == 'AUG') & (std5['HIGHZ'] == 1), 'TOK'] = 'AUGW'

    cols = [
        'TOK',
        'SHOT',
        'GROUP',
        'CONFIG',
        'IP', 
        'BT', 
        'NEL', 
        'RGEO', 
        'AMIN', 
        'KAREA', 
        'EPSILON',
        'DELTA',
        'DELTAU', 
        'DELTAL', 
        'MEFF',
        'POHM', 
        'PNBI', 
        'PECRH',
        'PICRH',
        'PLTH',
        'PRAD',
        'TAUTH', 
        'TAU98Y2', 
        'HIPB98Y2',
        'TAU20', 
        'H20',
        'IPN',
        'qCYL',
        'GWF',
        'HIGHZ', 
        'COAT',
        'PFIN',
        'PRFN',
        'PLH',
        'SELDB5',
        'STD3',
        'DB2P8',
        'AUXHEAT',
        'PHASE',
        'HYBRID',
        'ITB',
        'ITBTYPE',
        'ELMTYPE',
    ]

    std5 = std5[cols]
    std3 = std5[std5['STD3'] == True]
    print("After")
    print(len(std3))
    print(std3['TOK'].value_counts().sort_index())
    print(len(std5))
    print(std5['TOK'].value_counts().sort_index())

    # Count data where PNBIN + PRFN < 1
    count_lt_1 = len(std5[std5['PFIN'] + std5['PRFN'] < 1])
    print(f"Number of data points where PFIN + PRFN < 1: {count_lt_1}")


    # Save
    std5.to_csv('data/processed/normalized.csv', index=True)

if __name__ == '__main__':
    preprocess(data_path) 