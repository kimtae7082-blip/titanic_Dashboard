from utils.training import train_all

if __name__=='__main__':
    result=train_all()
    print(result['comparison'].round(3).to_string(index=False))
    print('CV 선택 모델:',result['selected'])
