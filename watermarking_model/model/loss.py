import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, train_config):
        super(Loss, self).__init__()
        self.msg_loss = nn.MSELoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        msg_loss = self.msg_loss(msg, rec_msg)
        return embedding_loss, msg_loss

class Loss_identity(nn.Module):
    def __init__(self, train_config):
        super(Loss_identity, self).__init__()
        self.msg_loss = nn.MSELoss()
        # self.msg_loss = nn.CrossEntropyLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1])
        return embedding_loss, msg_loss


class Loss_identity_3(nn.Module):
    def __init__(self, train_config):
        super(Loss_identity_3, self).__init__()
        self.msg_loss = nn.MSELoss()
        # self.msg_loss = nn.CrossEntropyLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        # msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1]) + self.msg_loss(msg, rec_msg[2])
        # msg_loss = self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3])
        msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1]) + self.msg_loss(msg, rec_msg[2]) + self.msg_loss(msg, rec_msg[3])
        return embedding_loss, msg_loss

class Loss_identity_3_2(nn.Module):
    def __init__(self, train_config):
        super(Loss_identity_3_2, self).__init__()
        # self.msg_loss = nn.MSELoss()
        self.msg_loss = nn.BCEWithLogitsLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        # msg_loss = self.msg_loss(msg, rec_msg[0]) + self.msg_loss(msg, rec_msg[1]) + self.msg_loss(msg, rec_msg[2])
        # msg_loss = self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3]) + self.msg_loss(msg, rec_msg[3])
        msg_loss =  self.msg_loss(rec_msg[0].squeeze(1), msg.squeeze(1)) + \
                    self.msg_loss(rec_msg[1].squeeze(1), msg.squeeze(1)) + \
                    self.msg_loss(rec_msg[2].squeeze(1), msg.squeeze(1)) + \
                    self.msg_loss(rec_msg[3].squeeze(1), msg.squeeze(1))
        return embedding_loss, msg_loss

class Loss2(nn.Module):
    def __init__(self, train_config):
        super(Loss2, self).__init__()
        # self.msg_loss = nn.MSELoss()
        self.msg_loss = nn.BCEWithLogitsLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        msg_loss = self.msg_loss(rec_msg.squeeze(1), msg.squeeze(1))
        return embedding_loss, msg_loss

class Loss_identity_2(nn.Module):
    def __init__(self, train_config):
        super(Loss_identity_2, self).__init__()
        self.msg_loss = nn.BCEWithLogitsLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg):
        embedding_loss = self.embedding_loss(x, w_x)
        msg_loss = self.msg_loss(rec_msg[0].squeeze(1), msg.squeeze(1)) + self.msg_loss(rec_msg[1].squeeze(1), msg.squeeze(1))
        return embedding_loss, msg_loss

class Lossex(nn.Module):
    def __init__(self, train_config):
        super(Lossex, self).__init__()
        self.msg_loss = nn.MSELoss()
        # self.msg_loss = nn.CrossEntropyLoss()
        self.embedding_loss = nn.MSELoss()
    
    def en_de_loss(self, x, w_x, msg, rec_msg, no_msg, no_decoded):
        embedding_loss = self.embedding_loss(x, w_x)
        msg_loss = self.msg_loss(msg, rec_msg)
        no_msg_loss = self.msg_loss(no_msg, no_decoded)
        return embedding_loss, msg_loss, no_msg_loss
