#========================= Python imports ========================
from __future__ import annotations
import os

from enum import Enum


#========================= Lib imports ========================

from tqdm import tqdm

import torch
import torch.nn as nn

from torchvision import transforms

#========================= My imports ========================

from SimpleUnetModel import SimpleUNetModel
from SegmentationInputLoader import SegmentationInputLoader, LoaderType
from Settings import Settings
from ImageUtils import ImageUtils, ToRGB, MyToTensor

#================================================================================
#================================================================================
#================================================================================

def train_one_epoch(epoch_index, 
                    model: nn.Module, 
                    loader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    loss_fn: nn.Module, 
                    s: Settings):
    running_loss = 0.
    count = 0
   
    model.train(True)

    tepoch = tqdm(loader, unit="batch",  desc=f"epoch: {epoch_index}")

    for data in tepoch:    
        # Every data instance is an input + label pair
        x, gt = data

        x = x.to(s.device, non_blocking=True)
        gt = gt.to(s.device, non_blocking=True)
               
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(x)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, gt)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        count += 1
       
        tepoch.set_postfix(loss=loss.item())

    return running_loss / count

#================================================================================

def run_test(epoch_index, 
             model: nn.Module, 
             loader: torch.utils.data.DataLoader, 
             loss_fn: nn.Module, 
             s: Settings):
    running_loss = 0.
    count = 0
   
    tepoch = tqdm(loader, unit="batch", desc=f"epoch: {epoch_index}")

    for data in tepoch:    
        # Every data instance is an input + label pair
        x, gt = data

        x = x.to(s.device, non_blocking=True)
        gt = gt.to(s.device, non_blocking=True)
                      
        # Make predictions for this batch
        outputs = model(x)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, gt)
        loss.backward()

        
        # Gather data and report
        running_loss += loss.item()
        count += 1
       
        tepoch.set_postfix(loss=loss.item())

    return running_loss / count

#================================================================================

def run_inference(model: nn.Module, 
                  s: Settings, 
                  imgPath: str, 
                  outputPath: str, 
                  threshold: bool = False):

    t = []        
    if (s.channelsCount == 1):            
        t.append(transforms.Grayscale())
    elif (s.channelsCount == 3): 
        t.append(ToRGB())

    t.append(transforms.Resize(size=(s.imgH, s.imgW), antialias=True))
               
    t.append(MyToTensor())
               
    t = transforms.Compose(t)

    x = ImageUtils.loadImageAsTensor(imgPath,
                                    imgFormat=ImageUtils.ImageFormat.C_H_W,
                                    transform=t)

    x = x.to(s.device, non_blocking=True)

    #model expects batch at input, we add batch of size 1 to tensor
    x = x.unsqueeze(0)

    output = model(x)
    
    output = torch.sigmoid(output)

    if (threshold):
        output[output > 0.5] = 1.
        output[output <= 0.5] = 0.

    ImageUtils.tensorToImage(output).save(outputPath)

#================================================================================

if __name__ == "__main__":

    s = Settings()

    print(f"Current dir: {s.currentDir}")
    print(f"Dataset path: {s.datasetPath}")
    print(f"Will run on {s.device}")

    #---------------------------------------------------------------------------
    #Prepare training data
    trainLoader = SegmentationInputLoader(s, loaderType=LoaderType.TRAIN, 
                                          subsetSize = s.maxTrainSize)
    trainDataset = trainLoader.buildDataLoader()

    testLoader = SegmentationInputLoader(s, loaderType=LoaderType.TEST, 
                                         subsetSize = s.maxTestSize)
    testDataset = testLoader.buildDataLoader()
    
    #---------------------------------------------------------------------------
    #Prepare model
    model = SimpleUNetModel(channels_in=s.channelsCount,
                            channels_out=1,
                            out_w=s.imgW,
                            out_h=s.imgH
                            )
    if (s.loadModelCheckpoint is not None):
        model.load_state_dict(torch.load(s.loadModelCheckpoint, weights_only=True))

    model = model.to(s.device)

    #test model without training just to see if all is running
    #if there is some problem, it will crash somewhere
    run_inference(model, 
                  s, 
                  os.path.join(s.currentDir, "test_data", "Lenna.png"), 
                  os.path.join(s.currentDir, "outputs", "Lenna_output_not_trained.png"))

    run_inference(model, 
                  s, 
                  os.path.join(s.currentDir, "test_data", "20120229_121714.jpg"), 
                  os.path.join(s.currentDir, "outputs", "20120229_121714_output_not_trained.png"))
    
    #---------------------------------------------------------------------------
    #Prepare training loss function and optimizer

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    #---------------------------------------------------------------------------
    # Training

    for epoch in range(s.epochCount):            
        avg_loss = train_one_epoch(epoch, model, trainDataset, optimizer, loss_fn, s)

        if (epoch % s.saveEveryNthEpoch == 0):
            torch.save(model.state_dict(), os.path.join(s.currentDir, "checkpoints", f"model_chackpoint_{epoch}.pth"))

    #---------------------------------------------------------------------------
    # Testing
    model.eval()
    avg_loss = run_test(epoch, model, trainDataset, loss_fn, s)

    #---------------------------------------------------------------------------
    # Running

    #use some image from "training" data    
    run_inference(model, 
                  s, 
                  os.path.join(s.datasetPath, "107252444", "20221124_1100.jpg"), 
                  os.path.join(s.currentDir, "outputs", "20221124_1100_output_trained.png"))

    #use some different image
    run_inference(model, 
                  s, 
                  os.path.join(s.currentDir, "test_data", "Lenna.png"), 
                  os.path.join(s.currentDir, "outputs", "Lenna_output_trained.png"))

    run_inference(model, 
                  s, 
                  os.path.join(s.currentDir, "test_data", "20120229_121714.jpg"), 
                  os.path.join(s.currentDir, "outputs", "20120229_121714_output_trained.png"))

    run_inference(model, 
                  s, 
                  os.path.join(s.currentDir, "test_data", "20120229_121714.jpg"), 
                  os.path.join(s.currentDir, "outputs", "20120229_121714_output_trained_threshold.png"),
                  True)


    #---------------------------------------------------------------------------
    # todo:
    # missing saving & loading trained model

    print("Finished")