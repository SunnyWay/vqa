require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'hdf5'
require 'rnn'
cjson=require 'cjson'
npy4th=require 'npy4th'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Training VQA model')
cmd:text()
cmd:text('Options')

cmd:option('-input_json', 'data/data.json', 'json data for training')
cmd:option('-input_h5', 'data/data.h5', 'h5 data for training')

cmd:option('-noeval', false, 'evaluate locally (used for split 2 and 3)')

cmd:option('-emb_size', 200, 'word embedding size')
cmd:option('-lstm_hs', 512, 'hidden size of lstm')
cmd:option('-lstm_nl', 2, 'number of lstm layer')
cmd:option('-cs', 1024, 'common size of question and image feature')

cmd:option('-lr', 3e-4, 'learning rate for rmsprop')
cmd:option('-lr_decay', 0.99997592083, 'lr decay factor in each iteration')
cmd:option('-batch_size', 512, 'batch size for each iteration')
cmd:option('-max_iter', 50000, 'max number of iteration')

cmd:option('-save_cp_every', 2000, 'frequency for saving model')
cmd:option('-cp_path', 'checkpoint', 'path to save checkpoints')

cmd:option('-test_every', 2000, 'frequency for testing')
cmd:option('-test_task', 'oe', 'which task test on, oe | mc | both')

cmd:option('-gpuid', 0, 'gpu no. to use')
cmd:option('-seed', 1234, 'random number generator seed to use')

opt = cmd:parse(arg)
timestamp = os.date('%Y%m%d_%H%M%S')
cp_path = opt.cp_path..'/'..timestamp
opt.rundir = cmd:string(cp_path, opt, {input_json=true, input_h5=true, max_iter=true})
paths.mkdir(opt.rundir)

cmd:log(opt.rundir..'/log', opt)
cmd:addTime('vqa', '%T')

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.manualSeed(opt.seed)
    cutorch.setDevice(opt.gpuid + 1)
end

print('')
print('** Load Data **')

do
    print('Load h5 file: ', opt.input_h5)
    local hf = hdf5.open(opt.input_h5, 'r')
    trainset = {}
    trainset['ques_id'] = hf:read('/train_ques_id'):all()
    trainset['txt'] = hf:read('/train_txt'):all()
    trainset['img_pos'] = hf:read('/train_img_pos'):all()
    trainset['ans'] = hf:read('/train_ans'):all()
    testset = {}
    testset['ques_id'] = hf:read('/test_ques_id'):all()
    testset['txt'] = hf:read('/test_txt'):all()
    testset['img_pos'] = hf:read('/test_img_pos'):all()
    testset['MC'] = hf:read('/test_MC'):all()
    hf:close()

    print('Load json file: ', opt.input_json)
    local fp = io.open(opt.input_json, 'r')
    local txt = fp:read()
    fp:close()
    code_book = cjson.decode(txt)

    -- Load image feature
    function load_imfea(file_list)
        local imfea = nil
        for _,fname in pairs(file_list) do
            print('Load image feature file: ', fname)
            local tmp = npy4th.loadnpy(fname)
            if imfea == nil then
                imfea = tmp
            else
                imfea = torch.cat(imfea, tmp, 1)
            end
        end
        return imfea
    end
    trainset['img_fea'] = load_imfea(code_book.train_img_fea_file)
    testset['img_fea'] = load_imfea(code_book.test_img_fea_file)
end


print('')
print('** Build Network **')

nword = #code_book.itow
nanswer = #code_book.itoa
seq_len = trainset['txt']:size(2)
img_fea_size = trainset['img_fea']:size(2)

do
    local txt_encoder = nn.Sequential()
        :add(nn.LookupTableMaskZero(nword, opt.emb_size))
        :add(nn.Dropout(0.5))
        :add(nn.Tanh())

    assert(opt.lstm_nl > 0)
    local in_size = opt.emb_size
    for i=1,opt.lstm_nl do
        local lstm = nn.SeqLSTM(in_size, opt.lstm_hs)
        lstm.batchfirst = true
        lstm.maskzero = true
        txt_encoder:add(lstm)
        txt_encoder:add(nn.Dropout(0.5))
        in_size = opt.lstm_hs
    end
    txt_encoder:add(nn.Select(2, seq_len))
        

    local txt = nn.Identity()()
    local img = nn.Identity()()

    local htxt = nn.Tanh()(nn.Linear(opt.lstm_hs, opt.cs)(txt_encoder(txt)))
    local himg =nn.Tanh()(nn.Linear(img_fea_size, opt.cs)(nn.Dropout(0.5)(img)))
    local hm = nn.CMulTable()({htxt, himg})

    local ans = nn.Linear(opt.cs, nanswer)(nn.Dropout(0.5)(hm))
    model = nn.gModule({txt, img}, {ans})
    
    criterion = nn.CrossEntropyCriterion()
end


print('')
print('** Train model **')

-- prepare dataset
local dataset = {}

function dataset:init(is_test)
    self.N = self['txt']:size(1)
    self.batch_idx = 1
    self.test = is_test
end

function dataset:next_batch()
    assert(self.batch_idx and self.batch_idx > 0)

    local bs = opt.batch_size
    local N = self.N
    
    local b = (self.batch_idx-1)*bs + 1
    local e = math.min(self.batch_idx*bs, N)

    local sid = self['ques_id'][{{b,e}}]
    local fv_txt = self['txt'][{{b,e}}]
    local img_pos = self['img_pos'][{{b,e}}]:long()
    local fv_img = self['img_fea']:index(1, img_pos)

    local mc = nil
    if self['MC'] ~= nil then
        mc = self['MC'][{{b,e}}]
    end

    local label = nil
    if self['ans'] ~= nil then
        label = self['ans'][{{b,e}}]
    end

    -- ship data to gpu
    if opt.gpuid >= 0 then
        fv_txt = fv_txt:cuda()
        fv_img = fv_img:cuda()
        if label ~= nil then
            label = label:cuda()
        end
    end

    -- move the pointer
    self.batch_idx = self.batch_idx + 1
    if self.batch_idx * bs > N+bs then
        self.batch_idx = 1

        -- shuffle samples
        if not self.test then
            qidx = torch.randperm(N):long()
            self['ques_id'] = self['ques_id']:index(1, qidx)
            self['txt'] = self['txt']:index(1, qidx)
            self['img_pos'] = self['img_pos']:index(1, qidx)

            if self['MC'] ~= nil then
                self['MC'] = self['MC']:index(1, qidx)
            end

            if self['ans'] ~= nil then
                self['ans'] = self['ans']:index(1, qidx)
            end
        end
    end

    return sid, fv_txt, fv_img, label, mc
end

setmetatable(trainset, {__index=dataset})
trainset:init(false)
setmetatable(testset, {__index=dataset})
testset:init(true)

-- initialize model parameters
if opt.gpuid >= 0 then
    print('shipped model to cuda...')
    model = model:cuda()
    criterion = criterion:cuda()
end

param, grad_param = model:getParameters()
param:uniform(-0.08, 0.08)

function JdJ(x)
    if param ~= x then
        param:copy(x)
    end

    grad_param:zero()

    local _, fv_txt, fv_img, label, _ = trainset:next_batch()
    local score = model:forward({fv_txt, fv_img})
    local f = criterion:forward(score, label)
    local dscore = criterion:backward(score, label)
    model:backward({fv_txt, fv_img}, dscore)

    grad_param:clamp(-10, 10)

    if running_avg == nil then
        running_avg = f
    end
    running_avg = running_avg*0.95+f*0.05

    return f, grad_param
end

function test()
    model:evaluate()
    criterion.nll.sizeAverage = false

    local N = testset.N
    local bs = opt.batch_size

    local oe_result = {}
    local mc_result = {}
    local itoa = code_book.itoa

    for i=1,N,bs do
        local sid, fv_txt, fv_img, label, mc = testset:next_batch()
        local score = model:forward({fv_txt, fv_img})

        if opt.test_task == 'oe' or opt.test_task == 'both' then
            local _, oe_ans_idx = score:max(2)
            for j=1,sid:size(1) do
                oe_result[i+j-1] = {question_id=sid[j], answer=itoa[oe_ans_idx[j][1]]}
            end
        end

        if opt.test_task == 'mc' or opt.test_task == 'both' then
            for j=1,mc:size(1) do
                local mc_score = {}
                local mc_score_pos = {}
                for k=1,mc:size(2) do
                    if mc[j][k] ~= 0 then
                        table.insert(mc_score_pos, mc[j][k])
                        table.insert(mc_score, score[j][mc[j][k]])
                    end
                end
                local _, idx = torch.Tensor(mc_score):max(1)
                mc_result[i+j-1] = {question_id=sid[j], answer=itoa[mc_score_pos[idx[1]]]}
            end
        end
    end

    local oe_txt = cjson.encode(oe_result)
    local mc_txt = cjson.encode(mc_result)
    collectgarbage()

    model:training()
    criterion.nll.sizeAverage = true
    return oe_txt, mc_txt
end

optimize = {}
optimize.learningRate = opt.lr
optimize.weightDecay = 0
optimize.winit = param

max_iter = opt.max_iter
cp_every = opt.save_cp_every
test_every = opt.test_every
state = {}

for iter=1,max_iter do
    if cp_every > 0 and iter%cp_every == 0 then
        torch.save(string.format('%s/iter_%d', opt.rundir, iter),{param})
    end
    
    if iter%100 == 0 then
        print(string.format('[%d/%d]training loss: %.6f\tlr: %.6f', 
            iter, max_iter, running_avg, optimize.learningRate))
    end

    if test_every > 0 and iter%test_every == 0 then
        local oe_txt, mc_txt = test()
        if opt.test_task == 'oe' or opt.test_task == 'both' then
            local fname = string.format('%s/oe_iter_%d_result.json', opt.rundir, iter)
            local fp = io.open(fname, 'w')
            fp:write(oe_txt)
            fp:close()
            if not opt.noeval then
                os.execute('python evaluate.py --res_file '..fname..' --task OpenEnded'..' > /dev/null')
            end
        end

        if opt.test_task == 'mc' or opt.test_task == 'both' then
            fname = string.format('%s/mc_iter_%d_result.json', opt.rundir, iter)
            fp = io.open(fname, 'w')
            fp:write(mc_txt)
            fp:close()
            if not opt.noeval then
                os.execute('python evaluate.py --res_file '..fname..' --task MultipleChoice'..' > /dev/null')
            end
        end
    end

    optim.rmsprop(JdJ, optimize.winit, optimize, state)
    optimize.learningRate = optimize.learningRate * opt.lr_decay

    if iter % 50 == 0 then
        collectgarbage()
    end
end

torch.save(string.format('%s/iter_%d', opt.rundir, max_iter),{param})
