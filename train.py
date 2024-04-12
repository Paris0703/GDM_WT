
from torch.utils.data import DataLoader

from netPaper3 import LightNet
from data_loader import *
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import LambdaLR


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
init_lr=0.00004
def lr_schedule_cosdecay(t, T, init_lr=init_lr, end_lr=0.000001):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dehazeNet = LightNet().to(device)
dehazeNet.load_state_dict(torch.load(r"I:\gxl\lightDehaze\weights_haze4k\19_0.7330, .pth"))


optimizer = torch.optim.Adam(dehazeNet.parameters(), lr =init_lr, betas=(0.9, 0.999),eps=1e-08)
dataloader = DataLoader(PairLoader(),batch_size=8, shuffle=False, num_workers=8,pin_memory=True,drop_last=True)


criterion = nn.MSELoss()

if __name__ == "__main__":
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    epoch = 0

    lossAll = 0
    losslow = 0
    lossmiddle = 0
    losshigh = 0
    for name, param in dehazeNet.named_parameters():
        print(f"{name}: {optimizer.param_groups[0]['lr']}")
        break
    for i, batch in loop:
        # Set model input
        with torch.no_grad():
            real_A = batch['clearimage'].to(device).detach()
            real_B = batch['hazeimage'].to(device).detach()
            transformInput1 = batch['transformInput1'].to(device).detach()
            transformInput2 = batch['transformInput2'].to(device).detach()
            transformInput3 = batch['transformInput3'].to(device).detach()
            mask = batch['mask'].to(device).detach()
            real_B = torch.cat([real_B,mask],dim=1)
            transformInput = torch.cat([transformInput1, transformInput2,transformInput3], dim=1)
            finalInput = torch.cat([real_B,transformInput],dim=1).detach()


        optimizer.zero_grad()
        fakeClear = dehazeNet(finalInput)
        # fakeClearFeature = criterionNet(fakeClear)
        # ClearFeature = criterionNet(real_A)

        lossSame = criterion(fakeClear,real_A)
        lossSame.backward()
        # lossFeature1 = criterion(fakeClearFeature[0],ClearFeature[0])
        # lossFeature2 = criterion(fakeClearFeature[1], ClearFeature[1])
        # lossFeature3 = criterion(fakeClearFeature[2], ClearFeature[2])

        #(lossSame+lossFeature1+lossFeature2+lossFeature3).backward()
        optimizer.step()

        lossAll = lossAll + lossSame.detach()
        # losslow = losslow + lossFeature1.detach()
        # lossmiddle = lossmiddle + lossFeature2.detach()
        # losshigh = losshigh + lossFeature3.detach()
        loop.set_description(f'[{epoch}/{100}]')
        loop.set_postfix(lossAll=lossAll,losslow=losslow,lossmiddle=lossmiddle,losshigh=losshigh)

        if (i % 1000 == 0):
            lr = lr_schedule_cosdecay(i, len(dataloader))
            # scheduler.step()
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            epoch = epoch + 1
            for name, param in dehazeNet.named_parameters():
                print(f"{name}: {optimizer.param_groups[0]['lr']}")
                break

            torch.save(dehazeNet.state_dict(),
                       r'I:\gxl\lightDehaze\weights_haze4k\\' + str(epoch) + "_" + str(lossAll)[
                                                                                              7:15] + '.pth')
            lossAll = 0
            losslow = 0
            lossmiddle = 0
            losshigh = 0