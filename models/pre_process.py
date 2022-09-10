
import pandas as pd
import numpy as np

#identifica as diferenças
def discrepancy_4(serie,n=5,reverse=False): 
    #Função que identifica diferenças entre duas amostras consecutivas de tamanho n
    #   se a diferença for maior que 3*amplitude, então é considerada uma discrepância
    
    # Além disso, compara a diferença de uma amostra com o ponto seguinte
    #   se a diferença for maior que 2, então é considerada uma discrepância
    
    #Os valores foram ajustados experimentalmente
    
    # n = tamanho da janela
    
    if reverse == True:
        serie = serie[::-1] 
    
    
    discrip = [0]*n # inicializa a lista de discrepância
    
    for i in range(len(serie)-n):
        sample1 = serie[i:i+n] # amostra 1
        min1 = np.min(sample1) # minimo da amostra 1
        max1 = np.max(sample1) # maximo da amostra 1
        
        
        sample2 = serie[i+n:i+2*n] # amostra 2
        min2 = np.min(sample2) # minimo da amostra 2
        max2 = np.max(sample2) # maximo da amostra 2
        
        ptp1 = 3*abs(max1-min1) # amplitude da amostra 1
        ptp2 = 3*abs(max2-min2) # amplitude da amostra 2
        
        if (min2>max1+ptp1 or max2<min1-ptp1): #amostra 2 está fora da amostra 1
            discrip.append(100) #discrepancia
            
        else: # amostra 2 está dentro da amostra 1
            if serie[i+n] > max1+2 or serie[i+n] < min1-2: # ponto atual está fora da amostra 1
                discrip.append(100) #discrepancia
            else:
                discrip.append(0) #sem discrepancia
        
        if reverse == True:
            return np.array(discrip[1,:].append(0))[::-1] 

    return np.array(discrip) #lista de discrepancias
        

# Pega as partes uteis da série
def loc_discrepancy(serie,disc1, l_train):
    #Recebe uma série, lista de discrepâncias e um intervalo de amostragem
    #Cada trecho entre discrepâncias é considerado uma nova série
    #Cada nova série recebe um idenficador
    
    #A função retorna uma série cujos trechos menores que o intervalo são substituidos por -1
    #Também retorna uma lista indicando o id de cada ponto da série
    samples = l_train + 10 # tamanho da janela
    d1 = 0
    remove_val = np.copy(serie) 
    parts = []
    part=1
    
    count = 0
    for i in range(0,len(disc1)): #percorre a lista de discrepâncias
        parts.append(part)
        
        if disc1[i]==100 or disc1[i]==200: #se a discrepância for 100 ou 200, o trecho é considerado uma nova série
            part+=1 #incrementa o identificador
            
            if i-d1<samples: #se o trecho for menor que o intervalo de amostragem, substitui por -1
                remove_val[d1:i]=[-1]*len(disc1[d1:i]) #substitui os valores por -1
                d1=i
            else: d1=i
            
    if i-d1<samples: #se o último trecho for menor que o intervalo de amostragem
        remove_val[d1:]=[-1]*len(disc1[d1:]) #substitui os valores por -1
        
    return remove_val, parts 
    #retorna a série com os trechos menores que o intervalo substituidos por -1 e a lista de identificadores
    
    
# Cria um novo banco de dado identificando os trechos das séries
def proces_data(data,l_train):
    fun_dis = discrepancy_4
    disc = []
    aenv= data.COD_AENV.unique()
    data_f = pd.DataFrame()
    for i in aenv:
        data_A=data[data['COD_AENV']==i].sort_values(by='DT_PROC')
        
        serie=data_A['VLR_DESG'].to_list()
        id = data_A['id'].to_list()
        
        disc1 = fun_dis(serie) + fun_dis(serie[::-1])[::-1]
        
        data_A['LOC_DISC'] = disc1
        
        serie,parts = loc_discrepancy(serie,disc1,l_train)
        
        data_A['VLR_DESG_new'] = serie
        data_A['SEC'] = parts
        
        #data_A = data_A[data_A.VLR_DESG_new!=-1]
        
        data_A['COD'] = range(0,len(data_A['SEC'].tolist()))
        
        data_f = pd.concat([data_f,data_A])    
        
    return data_f

# cria um banco de dados pronto para ser usado no modelo
def model_input(data_health, l_train=20,falh = 24):
    data = data_health[data_health.COD_FALH==falh]
    data_f = proces_data(data,l_train=20)

    l_test = 10 #tamanho da janela de teste (regra de negócio)
    n=l_train+l_test
    train_test = []
    treino = []
    teste = []
    id_trecho=[]
    aen_ = []

    for aenv in data_f.COD_AENV.unique():
        for i in data_f[data_f['COD_AENV']==aenv].SEC.unique():
            trecho = data_f[(data_f['COD_AENV']==aenv) & (data_f['SEC']==i)].VLR_DESG.tolist()
            if len(trecho)>n:
                for j in range(len(trecho)-n):
                    treino.append(trecho[j:j+l_train])
                    teste.append(trecho[j+l_train:j+l_train+l_test])
                    id_trecho.append(i)
                    aen_.append(aenv)
                    train_test.append(np.array([*trecho[j:j+l_train],*trecho[j+l_train:j+l_train+l_test],aenv]))
                    
    return np.array(train_test)