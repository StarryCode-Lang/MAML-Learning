import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse
import  time

from    meta import Meta

def evaluate_model(maml, db_train, device, args, sample_num=None, step=None):
    if sample_num is None:
        sample_num = 1000 // args.task_num
    if step is None:
        step = args.epoch - 1
    
    eval_start = time.perf_counter()
    accs = []
    for _ in range(sample_num):
        x_spt, y_spt, x_qry, y_qry = db_train.next('test')
        x_spt = torch.from_numpy(x_spt).pin_memory().to(device, non_blocking=True)
        y_spt = torch.from_numpy(y_spt).pin_memory().to(device, non_blocking=True)
        x_qry = torch.from_numpy(x_qry).pin_memory().to(device, non_blocking=True)
        y_qry = torch.from_numpy(y_qry).pin_memory().to(device, non_blocking=True)

        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
            test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
            accs.append(test_acc)

    accs = np.array(accs).mean(axis=0).astype(np.float16)
    eval_elapsed = time.perf_counter() - eval_start
    acc = accs[-1]
    
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f'checkpoints/maml_step{step}_acc{acc:.4f}.pth'
    torch.save({
        'step': step,
        'model_state_dict': maml.state_dict(),
        'accuracy': acc,
        'n_way': args.n_way,
        'k_shot': args.k_spt,
        'meta_lr': args.meta_lr,
        'update_lr': args.update_lr,
    }, save_path)
    
    return acc, accs, eval_elapsed, save_path


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

    saved_checkpoints = []

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
            print('step:', step,' [', ' '.join([f'{x:.4f}' for x in accs]), ']', f'\tavg_time/step(s): {round(avg_step_time, 4)}', f'\tthroughput(img/s): {round(images/avg_step_time, 1)}')
            tot_elapsed = 0.0
            steps_since_log = 0

        if step % 500 == 0:
            current_acc, accs, eval_elapsed, save_path = evaluate_model(
                maml, db_train, device, args, sample_num=1000//args.task_num, step=step
            )
            
            if not hasattr(maml, 'best_acc') or current_acc > maml.best_acc:
                maml.best_acc = current_acc
                saved_checkpoints.append({
                    'path': save_path,
                    'step': step,
                    'accuracy': current_acc
                })
                print(f'\n✓ 发现更好的模型: step={step}, acc={current_acc:.4f}\n')

            print('Test acc:','[',  ' '.join([f'{x:.4f}' for x in accs]), ']', f'\teval_time(s): {round(eval_elapsed, 2)}')
            
    # 训练结束后，进行最终评估
    print('\n--- 进行最终评估 ---')
    final_acc, final_accs, final_eval_elapsed, final_save_path = evaluate_model(
        maml, db_train, device, args, sample_num=200//args.task_num, step=args.epoch-1
    )
    print('最终评估 acc:', ' '.join([f'{x:.4f}' for x in final_accs]), f'\teval_time(s): {round(final_eval_elapsed, 2)}')
    
    saved_checkpoints.append({
        'path': final_save_path,
        'step': args.epoch - 1,
        'accuracy': final_acc
    })
    print(f'✓ 最终模型已保存: {final_save_path}\n')
    
    # 选择最佳模型并清理
    if saved_checkpoints:
        best_checkpoint = max(saved_checkpoints, key=lambda x: x['accuracy'])
        final_best_path = f'checkpoints/maml_best_final_acc{best_checkpoint["accuracy"]:.4f}.pth'
        
        import shutil
        shutil.copy(best_checkpoint['path'], final_best_path)
        
        for cp in saved_checkpoints:
            if os.path.exists(cp['path']):
                os.remove(cp['path'])
        
        print(f'✓ 训练完成！')
        print(f'✓ 最佳模型: step={best_checkpoint["step"]}, acc={best_checkpoint["accuracy"]:.4f}')
        print(f'✓ 保存位置: {final_best_path}')


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
