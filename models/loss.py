from common import *

#  https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
#  https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/4
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs        = F.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()


    def forward(self, logits, targets):

        probs = F.sigmoid(logits)
        num = targets.size(0)
        m1  = probs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1- score.sum()/num
        return score


def make_weight(labels_truth, pos_weight=0.5, neg_weight=0.5):
    B,C,H,W = labels_truth.size()
    weight = Variable(torch.FloatTensor(B*C*H*W)).cuda()

    pos = labels_truth.detach().sum()
    neg = B*C*H*W - pos
    if pos>0:
        pos_weight = pos_weight/pos.cpu().numpy()
        neg_weight = neg_weight/neg.cpu().numpy()
    else:
        pos_weight = 0
        neg_weight = 0

    weight[labels_truth.view(-1)> 0.5] = pos_weight
    weight[labels_truth.view(-1)<=0.5] = neg_weight

    weight = weight.view(B,C,H,W)
    return weight


##  http://geek.csdn.net/news/detail/126833
class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view (-1)
        t = labels.view (-1)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/(w.sum()+ 1e-12)
        return loss

class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num   = labels.size(0)
        w     = (weights).view(num,-1)
        w2    = w*w
        m1    = (probs  ).view(num,-1)
        m2    = (labels ).view(num,-1)
        intersection = (m1 * m2)
        score = 2. * ((w2*intersection).sum(1)+1e-12) / ((w2*m1).sum(1) + (w2*m2).sum(1)+1e-12)
        score = 1 - score.sum()/num
        return score


#https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
#https://github.com/unsky/focal-loss
class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.size_average = size_average


    def forward(self, logits, targets, class_weights=None, type='softmax'):
        targets = targets.view(-1, 1).long()

        if type=='sigmoid':
            if class_weights is None: class_weights =[0.5, 0.5]

            probs  = F.sigmoid(logits)
            probs  = probs.view(-1, 1)
            probs  = torch.cat((1-probs, probs), 1)
            selects = Variable(torch.FloatTensor(len(probs), 2).zero_()).cuda()
            selects.scatter_(1, targets, 1.)

        elif  type=='softmax':
            B,C,H,W = logits.size()
            if class_weights is None: class_weights =[1/C]*C

            logits  = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
            probs   = F.softmax(logits,1)
            selects = Variable(torch.FloatTensor(len(probs), C).zero_()).cuda()
            selects.scatter_(1, targets, 1.)

        class_weights = Variable(torch.FloatTensor(class_weights)).cuda().view(-1,1)
        weights = torch.gather(class_weights, 0, targets)


        probs      = (probs*selects).sum(1).view(-1,1)
        batch_loss = -weights*(torch.pow((1-probs), self.gamma))*probs.log()


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss



def dice_loss(m1, m2, is_average=True):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)
    scores = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    if is_average:
        score = scores.sum()/num
        return score
    else:
        return scores


def multi_loss(logits, labels):
    #l = BCELoss2d()(logits, labels)


    if 0:
        l = BCELoss2d()(logits, labels) + SoftDiceLoss()(logits, labels)

    #compute weights
    else:
        batch_size,C,H,W = labels.size()
        weights = Variable(torch.tensor.torch.ones(labels.size())).cuda()

        if 1: #use weights
            kernel_size = 5
            avg = F.avg_pool2d(labels,kernel_size=kernel_size,padding=kernel_size//2,stride=1)
            boundary = avg.ge(0.01) * avg.le(0.99)
            boundary = boundary.float()

            w0 = weights.sum()
            weights = weights + boundary*2
            w1 = weights.sum()
            weights = weights/w1*w0

        l = WeightedBCELoss2d()(logits, labels, weights) + \
            WeightedSoftDiceLoss()(logits, labels, weights)

    return l


# AngularMSE loss for Direction Networks
class AngularMSE(nn.Module):
    def __init__(self):
        super(AngularMSE, self).__init__()

    def forward(self, pred, gt, weight):
        B, C, H, W = gt.size()
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, C)
        weight = weight.view(-1, 1)

        pred = F.normalize(pred, dim=1)*0.999999
        gt = F.normalize(gt, dim=1)*0.999999

        # print('pred type:', pred.type())
        # print('gt type:', gt.type())

        error_angles = (pred * gt).sum(1, keepdim=True).acos()
        # print('weight type:', weight.type())
        angles_loss = torch.sum(torch.abs(error_angles*error_angles)*weight)

        # total loss / batch_size ???
        return angles_loss/B      


# # check #################################################################
# def run_check_focal_loss():
#     batch_size  = 64
#     num_classes = 15
#
#     logits = np.random.uniform(-2,2,size=(batch_size,num_classes))
#     labels = np.random.choice(num_classes,size=(batch_size))
#
#     logits = Variable(torch.from_numpy(logits)).cuda()
#     labels = Variable(torch.from_numpy(labels)).cuda()
#
#     focal_loss = FocalLoss(gamma = 2)
#     loss = focal_loss(logits, labels)
#     print (loss)
#
#
# def run_check_soft_cross_entropy_loss():
#     batch_size  = 64
#     num_classes = 15
#
#     logits = np.random.uniform(-2,2,size=(batch_size,num_classes))
#     soft_labels = np.random.uniform(-2,2,size=(batch_size,num_classes))
#
#     logits = Variable(torch.from_numpy(logits)).cuda()
#     soft_labels = Variable(torch.from_numpy(soft_labels)).cuda()
#     soft_labels = F.softmax(soft_labels,1)
#
#     soft_cross_entropy_loss = SoftCrossEntroyLoss()
#     loss = soft_cross_entropy_loss(logits, soft_labels)
#     print (loss)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_soft_cross_entropy_loss()

    print('\nsucess!')
