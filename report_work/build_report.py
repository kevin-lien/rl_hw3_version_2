"""Build figures and a filled copy of the course LaTeX report template."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'report_work'; TEX=WORK/'latex';FIG=TEX/'figures';FIG.mkdir(exist_ok=True)
def read(name):return json.loads((WORK/'results'/name).read_text())
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})

def figures():
    fig,axs=plt.subplots(1,3,figsize=(10,3.1))
    for ax,label,title in zip(axs,['dqn','ddqn','final_ddqn'],['DQN, 40k steps','DDQN, 40k steps','Submitted DDQN, 360k steps']):
        rows=read(f'mc_{label}.json')['states']; x=np.array([r['mc'] for r in rows]);y=np.array([r['predicted'] for r in rows]);err=[r['ci95_half'] for r in rows]
        ax.errorbar(x,y,xerr=err,fmt='o',capsize=2,markersize=4,color='#23758b')
        lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],'--',color='gray')
        ax.set(title=title,xlabel='Monte Carlo Q (95% CI)',ylabel='Predicted Q')
    fig.tight_layout();fig.savefig(FIG/'mc_q.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,2.7))
    for label in ['dqn','ddqn']:
        directory=next((WORK/'results/tb').glob(label+'_*'));events=EventAccumulator(str(directory));events.Reload()
        for ax,key in zip(axs,['train/q_mean','train/loss']):
            rows=events.Scalars(key);ax.plot([r.step for r in rows],[r.value for r in rows],label=label.upper())
            ax.set(xlabel='Environment steps',ylabel=key);ax.legend()
    fig.tight_layout();fig.savefig(FIG/'dqn_learning.png',dpi=160);plt.close(fig)
    hist=json.loads((ROOT/'astro_party/results/6o6gk5el/history.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(9,2.7))
    for opponent in ['rusher','sniper']:
        axs[0].plot([r['global_step'] for r in hist],[r[f'eval/{opponent}_win_rate'] for r in hist],'o-',label=opponent)
    axs[0].set(xlabel='PPO environment steps',ylabel='Validation win rate');axs[0].legend()
    ev=EventAccumulator(str(next((ROOT/'astro_party/runs').glob('6o6gk5el_*'))));ev.Reload()
    key='train/value_loss';rows=ev.Scalars(key)
    axs[1].plot([r.step for r in rows],[r.value for r in rows],color='#bd6b35');axs[1].set(xlabel='PPO environment steps',ylabel='Value loss')
    fig.tight_layout();fig.savefig(FIG/'astro_learning.png',dpi=170);plt.close(fig)
    comparison=read('astro_algorithm_comparison.json')
    assert len(comparison)==6,'All three paired runs must finish before report generation'
    fig,axs=plt.subplots(1,2,figsize=(9,2.7))
    for ax,opponent in zip(axs,['rusher','sniper']):
        for seed in [42,43,44]:
            values=[next(r for r in comparison if r['algorithm']==algo and r['seed']==seed)['metrics'][f'eval/{opponent}_win_rate'] for algo in ['PPO','A2C']]
            ax.plot([0,1],values,'o-',label=f'rollout seed {seed}')
        ax.set(xticks=[0,1],xticklabels=['PPO','A2C'],ylabel='Win rate',title=opponent.capitalize(),ylim=(.5,1.02));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'astro_algorithms.png',dpi=170);plt.close(fig)

def escaped(s):
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_')]:s=s.replace(a,b)
    return s

def build():
    figures()
    final=read('2048_final.json');comp=read('2048_algorithm_comparison.json');astro=read('astro_algorithm_comparison.json')
    bias=[]
    for label,title in [('dqn','DQN 40k'),('ddqn','DDQN 40k'),('final_ddqn','Submitted DDQN')]:
        d=read(f'mc_{label}.json');score=final['evaluation'] if label=='final_ddqn' else comp[label]['evaluation']
        bias.append(f"{title} & {d['mean_bias']:.1f} & {d['mean_absolute_gap']:.1f} & {score['average_score']:.2f} & {score['best']} " + chr(92)*2)
    paired=[]
    for seed in [42,43,44]:
        for algo in ['PPO','A2C']:
            r=next(r for r in astro if r['algorithm']==algo and r['seed']==seed)['metrics']
            paired.append(f"{seed} & {algo} & {r['eval/rusher_win_rate']*100:.1f}\\% & {r['eval/sniper_win_rate']*100:.1f}\\% " + chr(92)*2)
    means={}
    for algo in ['PPO','A2C']:
        means[algo]=[np.mean([r['metrics'][f'eval/{op}_win_rate'] for r in astro if r['algorithm']==algo])*100 for op in ['rusher','sniper']]
    text=r'''\documentclass[a4paper]{article}
\usepackage{student}
\usepackage{float,graphicx,amsmath,amssymb,booktabs,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue}
\date{November 1, 2026}
\setmodule{Assignment 3 Report}
\setterm{115-1 (Fall 2026)}
\title{Reinforcement Learning}
\setmembername{Bo-Yu Lien}
\setmemberuid{b13901072}
\newcommand{\pic}[2]{\begin{center}\includegraphics[width=#1\linewidth]{figures/#2}\end{center}}
\begin{document}
\header{}
\section{2048}
\begin{answer}[Q1. Monte Carlo evaluation of $Q^\pi(s,a)$ (3 pts.)]
\textbf{What I estimate}
\begin{itemize}
\item The policy is greedy: $\pi(s)=\arg\max_a Q(s,a)$. I do not replace an illegal action with another action.
\item I collect 12 states from this policy's own games, using seeds starting from 71000. I take states at steps 0, 5, 15 and 30, when available. A state includes the board and the consecutive illegal-move count.
\item For each state, I fix the first action to $a=\pi(s)$. Then I follow the same greedy policy until the episode ends. I repeat it 64 times with independent future random seeds.
\end{itemize}
\[
 G_j=\sum_{t=0}^{T_j-1}\gamma^t r_t,\qquad
 \widehat Q^\pi(s,a)=\frac{1}{64}\sum_{j=1}^{64}G_j,\qquad \gamma=0.99.
\]
\textbf{Tile randomness and confidence interval}
\begin{itemize}
\item Every rollout starts from the same saved board, score and foul count. Only the future random-number stream changes. The environment still puts a 2 with probability 0.9 or a 4 with probability 0.1 into a random empty cell.
\item The seed is $800000+1000i+j$, where $i$ is the state index and $j$ is the rollout index. The initial board is restored after resetting the random generator.
\item I use a pointwise 95\% Student-$t$ interval: $\bar G\pm 1.99834\,s_G/\sqrt{64}$. It measures rollout randomness for one fixed state. It does not measure uncertainty across all possible boards or training runs.
\item There is a 2000-step safety cap. None of the measured rollouts reached this cap, so these returns did not need bootstrapping.
\end{itemize}
\textbf{Important detail}
\begin{itemize}
\item These Q values use the submitted \emph{training} reward and termination rule: merge score plus board shaping, illegal penalty $-300$, and termination after three consecutive illegal moves. This is needed to compare with the learned Q network.
\item The official evaluation has no shaping and ends at the first illegal move. I measure official score separately. A training Q value is not the expected official score.
\item Checkpoints do not save the old environment source. This check uses the current training environment; it cannot prove that every earlier experiment used exactly the same reward file.
\end{itemize}
\end{answer}
\newpage
\begin{answer}[Q2. Predicted Q vs. Monte Carlo Q for DQN and DDQN (5 pts.)]
\pic{1}{mc_q.png}
The horizontal bars are 95\% MC confidence intervals. The dashed line means the two values are equal. Each panel contains 12 states visited by that panel's greedy policy.
\begin{center}\small
\begin{tabular}{lrrrr}\toprule
Model & Mean $Q-\widehat Q$ & Mean absolute gap & Mean score & Best tile\\\midrule
@BIAS@
\bottomrule\end{tabular}\end{center}
\begin{itemize}
\item For a clean short-budget check, I train DQN and DDQN for 40,000 environment steps each. Both use seed 42, four environments, the same reward, network, replay settings and exploration schedule. Only the next-state target changes. The submitted DDQN is shown separately because it was trained longer.
\item DQN uses $y=r+\gamma\max_a Q_{\rm target}(s',a)$. The same target network selects and evaluates the action. A noisy large value can be selected by the maximum.
\item DDQN first selects $a^*=\arg\max_a Q_{\rm online}(s',a)$, then uses $y=r+\gamma Q_{\rm target}(s',a^*)$. This separates selection from evaluation. In code, I use \texttt{gather} to get the target value at the online network's chosen index.
\item Both short runs overestimate these sampled states. DDQN has a somewhat smaller mean gap in this run, but the states are different because each policy visits different boards. This small one-seed test is not strong evidence that DDQN always has lower bias.
\item The final DDQN has a smaller absolute gap here, and its mean signed gap is negative. Therefore, overestimation is not the only error. More training and a different state distribution also matter.
\item A small gap does not mean a good policy. A policy that stops early can correctly predict a low return. Also, shaped reward can be predicted well while the highest tile is still poor. I check actual tiles with the original evaluator.
\end{itemize}
\pic{.78}{dqn_learning.png}
Q means and losses from the matched 40k-step runs. The replay samples change over time, so this Q-mean plot is a training diagnostic, not a fixed-state calibration test.
\end{answer}
\newpage
\begin{answer}[Q3. Training the submitted 2048 model (4 pts.)]
\textbf{Final choice and network}
\begin{itemize}
\item I submit the weights from \texttt{model\_2048\_weight1000.zip}, from run \texttt{ddqn\_weight1000}. I rename a copy to \texttt{model\_2048.zip}. The run name is a label; it does not mean the current reward has a coefficient of 1000.
\item The model is Double DQN with SB3's built-in MLP policy. Input is the raw $16\times4\times4$ one-hot board. The flattened 256 inputs go through two 256-unit ReLU layers, then four Q outputs. The online network has 132,612 parameters, below the 5-million limit.
\end{itemize}
\begin{center}\small\begin{tabular}{ll}\toprule
Setting & Value\\\midrule
Learning rate / discount & $10^{-4}$ / 0.99\\
Replay size / batch size & 100,000 / 256\\
Learning starts / target update & 100 / 10,000 steps\\
Train frequency / gradient steps & every 4 vector steps / 1 update\\
Exploration & $\epsilon:1\to0.05$; exploration fraction 0.1\\
Training budget & 20 calls $\times$ 20,000 steps; four environments\\
Saved checkpoint & 360,000 steps, selected before the final 400k step\\
\bottomrule\end{tabular}\end{center}
\textbf{Reward and termination}
\begin{itemize}
\item A legal move gets its merge score plus $\Phi(B_{\rm after})-\Phi(B_{\rm before})$, where $\Phi(B)=\sum_{x,y}W_{x,y}B_{x,y}$. The after-board includes the new random tile.
\item The current weight matrix is below. It encourages larger tiles near the lower-left side, so the board has more order. This is ordinary reward shaping; with $\gamma<1$, it is not a policy-invariant potential-based guarantee.
\end{itemize}
\[
W=\begin{bmatrix}0&0&0&0\\0.5&0&0&0\\0.75&0.5&0&0\\1&0.75&0.5&0\end{bmatrix}.
\]
\begin{itemize}
\item An illegal move gives $-300$. Training allows up to three consecutive illegal moves, so exploration can try another direction. A legal move resets this counter. Official testing ends at the first illegal move, which is harder.
\item After each 20k-step call, the script checks ten games and saves when mean score or mean highest tile improves. The saved file has 360k steps. These training checks were not strictly fixed-seed checks because the current reset code reseeds on reset.
\item The old run did not set a training seed. The script reproduces the recipe, but exact old weights are not guaranteed. The new 40k-step comparison uses seed 42 and is not the submitted model.
\end{itemize}
\textbf{Official public test, seeds 0--99}
\begin{itemize}
\item Average score is 1238.0. Highest-tile counts are: 4:4, 8:2, 16:4, 32:10, 64:23, 128:36, 256:19, 512:2.
\item At least one game reaches 128, 256 and 512, so the original evaluator gives 15/15 public baseline points. This does not guarantee the private seeds.
\end{itemize}
\end{answer}
\newpage
\section{Astro Party}
\begin{answer}[Q4. Pipeline for the best performance (9 pts.)]
\textbf{Main idea: a rule-guided policy, then imitation and PPO}
\begin{itemize}
\item I first read the two bots and the observation layout. Rusher moves close quickly. Sniper aims from farther away. Destroying a ship is not enough: its pilot must also be killed before respawn.
\item My final method is a hybrid. A deterministic defensive controller proposes turn, thrust and shoot values from the current observation. A learned SB3 actor uses this proposal and ship information. This is not an end-to-end policy that discovered every tactic by itself.
\item The raw input is still the original 84 numbers. The custom extractor is defined inside the submitted training script, so it is saved with the model. No observation wrapper or normalization file is needed.
\end{itemize}
\textbf{Controller logic, in execution order}
\begin{enumerate}
\item Decode position, relative velocity, ammo, pilot status and observed bullets. Convert each coordinate with its own scale: horizontal 600 and vertical 400.
\item If my ship is a pilot, flee and survive. If the opponent is a pilot, hunt it. Otherwise, keep a useful fighting distance, roughly 500--700 pixels.
\item Aim ahead of the moving enemy, using bullet speed 620 and a velocity lead factor of 0.8. Check line of sight before deciding to fire.
\item Before the final action, simulate bullet distance for the next 60 steps. If the proposed move is unsafe, search 5 turn values $\times$ 3 thrust values. Prefer a safer move while avoiding unnecessary turns.
\item Recheck the firing direction after the chosen turn. This avoids shooting at the old heading while the ship is turning. Fire also needs ammo and cooldown readiness.
\end{enumerate}
\textbf{Policy and learning}
\begin{itemize}
\item The extractor returns the 3 controller outputs plus the first 26 observation entries: 29 features. The actor is a learned linear layer with three outputs. The value network has two 64-unit Tanh layers. Total trainable parameters: 6,238, below the 2-million limit.
\item I collect 10,000 teacher examples, then use 120 epochs of behavior cloning. The loss is mean squared error between the actor's mean action and the teacher action. Adam learning rate is 0.003. This gives a useful starting policy.
\item I then run real PPO training for $5\times8192=40960$ environment steps. Opponents are chosen 50/50 between Rusher and Sniper at episode reset. Spawns remain random.
\item PPO uses four environments, rollout length 512 per environment, batch size 256, five optimizer epochs, learning rate $10^{-6}$, $\gamma=0.995$, GAE $\lambda=0.95$, clip range 0.1, entropy coefficient 0 and initial log standard deviation $-3.5$.
\item Seed is 42. The small learning rate and small action noise help preserve aiming and dodging from imitation. The final exported model is the last 40,960-step checkpoint, not the validation-best file.
\end{itemize}
\end{answer}
\newpage
\begin{answer}[Q4. Reward, evidence and failed ideas (continued)]
\textbf{Final reward: the original defense profile}
\begin{center}\small\begin{tabular}{ll}\toprule
Event or progress & Reward\\\midrule
Final kill / being killed & $+20$ / $-20$\\
Enemy ejection / own ejection & $+3$ / $-3$\\
Pickup / each step & $+0.1$ / $-0.002$\\
Respawn / asteroid hit & 0 / 0\\
Distance progress & $c(d_{\rm previous}-d_{\rm now})$\\
Coefficient $c$ & own pilot: $-0.5$; enemy pilot: 2; otherwise: 0.5\\
\bottomrule\end{tabular}\end{center}
Distance is divided by 600. There is no velocity bonus in the final defense profile. Respawn has no positive reward, so repeatedly losing the ship is not useful by itself.
\pic{.98}{astro_learning.png}
Original reference run: validation seeds 10000--10049, 50 games per opponent. The second plot is PPO critic loss. A low critic loss alone is not evidence of winning.
\begin{center}\small\begin{tabular}{lrrl}\toprule
Test set & Rusher & Sniper & Games per opponent\\\midrule
Public seeds 0--99 & 95\% & 91\% & 100\\
Holdout seeds 50000--50099 & 92\% & 87\% & 100\\
Validation seeds 10000--10099 & 90\% & 84\% & 100\\
\bottomrule\end{tabular}\end{center}
\begin{itemize}
\item The public evaluator gives 10/10 baseline points. Draws count as failures. The public set was used during development, so I do not call it an untouched test set. The validation result of 84\% vs. Sniper also shows that passing is not guaranteed for every seed set.
\item I tested hiding behind the central planet, peeking to shoot, and fast same-direction escape. The fast version increased Sniper-game ship speed from 128.5 to 166.2 pixels/s, but on seeds 65000--65099 it scored only 72\% vs. Rusher and 80\% vs. Sniper. It was not selected.
\item A matched fast-controller test without velocity reward was already equally fast. Thus, the new navigation was the main speed improvement; extra speed reward did not show an extra speed benefit. Faster movement did not mean a better final policy.
\item My fresh bare-command retraining gives exactly the same policy tensors as the passed reference. ZIP bytes can differ because saved source and metadata differ. Fresh stock evaluation also gives 95\% and 91\% on the actual hand-in copy.
\end{itemize}
\end{answer}
\newpage
\begin{answer}[Q5. PPO vs. A2C: hypothesis and controlled experiment (9 pts.)]
\textbf{Hypothesis}
\begin{itemize}
\item A good imitation policy can lose useful behavior after noisy updates. My hypothesis is that PPO's clipped updates and reuse of each rollout will preserve the defensive policy better than A2C, especially against Sniper.
\item This experiment tests short-budget fine-tuning from the same BC policy. It does not test which algorithm learns this game best from random initialization.
\end{itemize}
\textbf{What is kept the same}
\begin{itemize}
\item Each pair starts with exactly the same actor and critic weights from the 10k-example BC checkpoint. Tensor equality is checked before training. Both use the same extractor, action noise, defense controller and original reward.
\item Both get 8192 new environment steps, four environments, rollout length 512, learning rate $10^{-6}$, $\gamma=0.995$, GAE $\lambda=0.95$, entropy coefficient 0, and normalized advantages. Both use Adam.
\item I repeat with rollout seeds 42, 43 and 44. Each algorithm pair uses matching environment seeds. These runs share the BC checkpoint, so they are three rollout-seed repeats, not three fully independent BC training runs.
\item Evaluation uses the same held-out seeds 72000--72029: 30 games against each bot per run. There is no model selection on this evaluation set. PPO keeps its five epochs, batch size 256 and clip range 0.1; A2C makes one update per rollout.
\end{itemize}
\begin{center}\small\begin{tabular}{llrr}\toprule
Rollout seed & Algorithm & Rusher win rate & Sniper win rate\\\midrule
@PAIRED@
\bottomrule\end{tabular}\end{center}
\pic{.98}{astro_algorithms.png}
Each line connects a matched pair with the same rollout seed. Games in the same seed set are not independent new test sets across the three runs.
\end{answer}
\newpage
\begin{answer}[Q5. Interpretation and limits (continued)]
\begin{itemize}
\item Mean Rusher win rates are @PPO_R@\% for PPO and @A2C_R@\% for A2C. Mean Sniper win rates are @PPO_S@\% and @A2C_S@\%. This is a descriptive average across the three rollout seeds.
\item @INTERPRET@
\item PPO clips policy-ratio updates and reuses each rollout for five epochs. A2C updates once without clipping. This may help PPO keep useful aiming behavior.
\item This is an equal-environment-step comparison, not equal optimizer work or wall time: PPO uses more minibatch updates. I cannot isolate clipping from sample reuse with this one experiment. A2C might need a different learning rate or more training.
\item Thirty games per opponent is a small sample. All pairs share BC data and test seeds. This does not establish a universal winner.
\item The full final model uses 40,960 PPO steps. These 8192-step experiment models are only for Q5 and do not replace the submitted model.
\end{itemize}
\textbf{Reproduce and evaluate}
\begin{itemize}
\item In the original Astro Party game folder, run \texttt{python train\_astro\_party.py}. It starts from teacher data and random network initialization, with no old checkpoint required. Output: \texttt{models/model\_astro\_party.zip}.
\item In the original 2048 game folder, run \texttt{python train\_2048.py}. The submitted configuration is the \texttt{ddqn\_weight1000} recipe with standard output name \texttt{models/model\_2048.zip}. Its original seed was not fixed, so this is approximate recipe reproduction.
\item Evaluate with the original scripts: \texttt{python eval\_astro\_party.py --model models/model\_astro\_party.zip} and \texttt{python eval\_2048.py --model models/model\_2048.zip}.
\item Both scripts support the flat hand-in and original \texttt{envs/} layout. Run them inside the original game folders, with the TA's game files and bots present.
\item Verified runtime: Python 3.11, CPU PyTorch 2.9.1 and SB3 2.9.0. The handout suggests Python 3.10; exact cross-version tensor reproduction has not been tested.
\end{itemize}
\textbf{Logs, references and assistance}
\begin{itemize}
\item Astro Party scalar training logs are on W\&B, project \texttt{astro-party-hw3}. Reference run: \texttt{6o6gk5el}; clean bare-command reproduction: \texttt{ghme0u0w}. Training records reward, PPO losses and validation wins. Q5 run: \texttt{znamd5vr}; its scalar record is @SYNC@.
\item Local TensorBoard logs, MC returns and analysis scripts are kept outside the seven-file hand-in folder.
\item Sources: assignment README, Assignment 3 slides (report p.49, policy p.51, submission p.53), provided report template, original game and bot code, SB3 DQN/PPO/A2C implementations, and the Double DQN paper by van Hasselt et al. (2015), \url{https://arxiv.org/abs/1509.06461}.
\item AI assistance was used for code, experiments and drafting. Numerical results come from saved logs. The text needs personal review under the course's report policy.
\end{itemize}
\end{answer}
\end{document}
'''
    interpretation='PPO has a higher Sniper win rate in each matched pair. This supports the hypothesis in this small warm-start experiment, but does not prove the mechanism.' if all(next(r for r in astro if r['algorithm']=='PPO' and r['seed']==s)['metrics']['eval/sniper_win_rate']>next(r for r in astro if r['algorithm']=='A2C' and r['seed']==s)['metrics']['eval/sniper_win_rate'] for s in [42,43,44]) else 'The paired results do not consistently favor PPO against Sniper. This does not support a clear advantage from the hypothesis under this short training budget.'
    replacements={'@BIAS@':'\n'.join(bias),'@PAIRED@':'\n'.join(paired),'@PPO_R@':f"{means['PPO'][0]:.1f}",'@A2C_R@':f"{means['A2C'][0]:.1f}",'@PPO_S@':f"{means['PPO'][1]:.1f}",'@A2C_S@':f"{means['A2C'][1]:.1f}",'@INTERPRET@':interpretation,'@SYNC@':'synced to the same W\&B project (scalar metrics and settings only)'}
    for key,value in replacements.items():text=text.replace(key,value)
    (TEX/'main.tex').write_text(text,encoding='utf-8')
    print('LaTeX report and four figures generated.')

if __name__=='__main__':build()
