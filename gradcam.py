class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        o = torch.sigmoid(output)
        o = o.cpu().data.numpy()
        if index == None:
            o = np.where(o>0.5, 1., 0.)
        label = o.sum(axis = 0)
        #print(label)
        label = np.where(label>o.shape[0]/2, 1., 0.)
        cam_list = []
        for i in range(len(label)):
            if label[i] == 0:
                cam_list.append(0)
                continue
                
            one_hot = np.zeros_like(label)
            one_hot[i] = 1.
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            if self.cuda:
                one_hot = torch.sum(one_hot.cuda() * output)
            else:
                one_hot = torch.sum(one_hot * output)
            
            self.feature_module.zero_grad()
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

            target = features[-1]
            target = target.cpu().data.numpy() # 40,2048,16,16

            weights = np.mean(grads_val, axis=(2, 3)) #40,2048
            weights = weights[:,:,np.newaxis,np.newaxis] #40,2048,1,1
            cam = np.zeros((target.shape[0], target.shape[2], target.shape[3]), dtype=np.float32) #40,16,16
            target =  weights * target #40,2048,16,16
            target = target.sum(axis=1) #40,16,16
            target = np.maximum(target, 0)
            T = np.zeros((input.shape[0],input.shape[2],input.shape[3]))
            for b in range(input.shape[0]):
                cam = cv2.resize(target[b], input.shape[2:])
                cam = cam - np.min(cam)
                cam = cam / np.max(cam)
                T[b] = cam
            cam_list.append(T)
    
        return label, cam_list
