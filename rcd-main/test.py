
import gen_data
import rcd
import utils as u

SEED = 42
NODES = 10
MIN_DEGREE = 1
MAX_DEGREE = 3
ANOMALOUS_NODES = 1
NORMAL_SAMPLES = 1_000
ANOMALOUS_SAMPLES = 1_000
STATES = 6

if __name__ == '__main__':
    tops=[0]*5
    for k in range(1000):
        src_dir, fe_service, an_nodes = gen_data.generate_data(k, NODES, MAX_DEGREE, NORMAL_SAMPLES, ANOMALOUS_SAMPLES, ANOMALOUS_NODES, STATES, False)
        (normal_df, anomalous_df) = u.load_datasets(src_dir + 'normal.csv', src_dir + 'anomalous.csv')
        result = rcd.top_k_rc(normal_df, anomalous_df, k=5, bins=None, localized=True)
        # if result['root_cause'][0] == an_nodes[0]:
        #     print('yes')

        rc = result['root_cause']
        for i in range(5):
            if i < len(rc) and rc[i] == an_nodes[0]:
                for j in range(i, 5):
                    tops[j] += 1
                break
    
    for i in range(5):
        print('TOP '+str(i+1)+': '+str(tops[i]/1000))
            
# TOP 1: 0.693
# TOP 2: 0.77
# TOP 3: 0.779
# TOP 4: 0.781
# TOP 5: 0.782

# 修改后没有明显提升
# TOP 1: 0.698
# TOP 2: 0.774
# TOP 3: 0.792
# TOP 4: 0.798
# TOP 5: 0.8