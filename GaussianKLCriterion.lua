-- A criterion for computing the KL-divergence between a Gaussian and N(0,I)
-- See Kingma and Welling (http://arxiv.org/pdf/1312.6114v10.pdf) Appendix B

require 'nn'
require 'torch'

local GaussianKLCriterion, parent = torch.class('nn.GaussianKLCriterion', 
                                                'nn.Criterion')
function GaussianKLCriterion:__init()
   parent.__init(self)
   self.gradInput = {}
end

function GaussianKLCriterion:updateOutput(input)
   local mean = input[1]
   local variance = input[2]
   
   if not self.divergences then
      self.divergences = torch.log(variance):add(1)
   else
      self.divergences:copy(variance)
      self.divergences:log():add(1)      
   end

   self.divergences:add(-1, torch.pow(mean, 2))
   self.divergences:add(-1, variance)
   
   return self.divergences:sum() * 0.5
end

function GaussianKLCriterion:updateGradInput(input, _)
   
   if #self.gradInput == 0 or self.gradInput[1]:size() ~= input[1]:size() then
      self.gradInput[1] = input[1]:clone():mul(-1)
      self.gradInput[2] = torch.ones(input[2]:size()) * -0.5
   else
      self.gradInput[1]:copy(input[1]):mul(-1)
      self.gradInput[2]:ones():mul(-0.5)
   end
   
   -- And the log term for the variance
   self.gradInput[2]:add(0.5, torch.ones(input[2]:size()):cdiv(input[2]))
   
   return self.gradInput
end
