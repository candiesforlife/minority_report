from datetime import datetime

def group_by_hour(df, year, month, day):
    '''
    get a sample of a month-time crimes grouped by hour
    inputs = start_date info
    '''
    sample = df.data[['period', 'latitude', 'longitude']]

    inf = sample['period'] > datetime(year, month, day, 0, 0, 0)
    sup = sample['period'] < datetime(year, month+1, day, 0, 0, 0)

    sample = sample[ inf & sup ]
    liste = np.sort(np.array(sample['period'].unique()))

    grouped = []
    length = len(liste)
    for index, timestamp in enumerate(liste):
        if (index+1) % 10 ==0:
            print(f'Grouping timestamp {index+1}/{length}')
        by_hour = np.array(sample[sample['period']== timestamp])
        grouped.append(by_hour)

    gr_array = np.array(grouped, dtype=object)
    len(gr_array[0])
    return gr_array
