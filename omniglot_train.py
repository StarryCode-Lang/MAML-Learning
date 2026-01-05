import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse
import  time

from    meta import Meta

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(args)

    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)

    log_interval = 100
    steps_since_log = 0
    tot_elapsed = 0.0

    for step in range(args.epoch):

        step_start = time.perf_counter()
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt = torch.from_numpy(x_spt).pin_memory().to(device, non_blocking=True)
        y_spt = torch.from_numpy(y_spt).pin_memory().to(device, non_blocking=True)
        x_qry = torch.from_numpy(x_qry).pin_memory().to(device, non_blocking=True)
        y_qry = torch.from_numpy(y_qry).pin_memory().to(device, non_blocking=True)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(x_spt, y_spt, x_qry, y_qry)

        step_elapsed = time.perf_counter() - step_start
        tot_elapsed += step_elapsed
        steps_since_log += 1

        if step % log_interval == 0:
            avg_step_time = tot_elapsed / max(steps_since_log, 1)
            setsz = x_spt.size(1)
            querysz = x_qry.size(1)
            images = (setsz + querysz) * args.task_num
            print('step:', step, '\ttraining acc:', accs, '\tavg_time/step(s):', round(avg_step_time, 4), '\tthroughput(img/s):', round(images/avg_step_time, 1))
            tot_elapsed = 0.0
            steps_since_log = 0

        if step % 1000 == 0:
            eval_start = time.perf_counter()
            accs = []
            for _ in range(1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt = torch.from_numpy(x_spt).pin_memory().to(device, non_blocking=True)
                y_spt = torch.from_numpy(y_spt).pin_memory().to(device, non_blocking=True)
                x_qry = torch.from_numpy(x_qry).pin_memory().to(device, non_blocking=True)
                y_qry = torch.from_numpy(y_qry).pin_memory().to(device, non_blocking=True)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc)

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            eval_elapsed = time.perf_counter() - eval_start
            print('Test acc:', accs, '\teval_time(s):', round(eval_elapsed, 2))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main(args)
