"""
plot_results.py

Loads all experiment JSON files and produces comparison plots.
Also prints the key diagnostic table to help interpret why hybrid succeeded or failed.

Usage:
  python plot_results.py                          # plots everything it finds
  python plot_results.py --files r1.json r2.json  # specific files

What to look for:
  1. Does hybrid ever cross above LLM?
     - If yes at any point: the IW fix is working, may need more time
     - If never: LinUCB may not be strong enough for this task

  2. Does p_llm (in corral results) decay over time?
     - Yes, decaying: CORRAL is learning that CB is getting better
     - Stays at p_min: CORRAL thinks LLM is always better than CB

  3. "CB acc when selected" vs "LLM acc when selected"
     - If CB acc < LLM acc throughout: LinUCB never learned well enough
     - If CB acc > LLM acc after warmup: hybrid should eventually dominate

  4. Compare hybrid to cold LinUCB:
     - If hybrid < cold LinUCB: IW is hurting, or LLM guidance is bad
     - If hybrid > cold LinUCB: warm-starting is working as intended
"""

import json
import glob
import argparse
import os


def load_results(pattern="results_*.json"):
    files = glob.glob(pattern)
    results = {}
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        name = os.path.splitext(os.path.basename(fp))[0]
        results[name] = data
    return results


def print_diagnostics(name, data):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    batches = data.get('batch', [])
    if not batches:
        print("  (no data)")
        return

    # Final values
    final_hybrid   = data['hybrid'][-1]   if data.get('hybrid')       else None
    final_llm      = data['llm'][-1]      if data.get('llm')           else None
    final_cold     = data['linucb_cold'][-1] if data.get('linucb_cold') else None
    final_p_llm    = data['p_llm'][-1]    if data.get('p_llm')         else data.get('p_llm_actual', [None])[-1]

    llm_sel = data.get('llm_acc_when_selected', [])
    cb_sel  = data.get('cb_acc_when_selected',  [])

    print(f"  Final batch:       {batches[-1]}")
    if final_hybrid is not None:
        print(f"  Hybrid reward:     {final_hybrid:.4f}")
    if final_llm is not None:
        print(f"  LLM reward:        {final_llm:.4f}")
    if final_cold is not None:
        print(f"  Cold LinUCB:       {final_cold:.4f}")
    if final_p_llm is not None:
        print(f"  Final p_llm:       {final_p_llm:.3f}")

    if data.get('llm_picks'):
        total_llm = data['llm_picks'][-1]
        total_cb  = data['cb_picks'][-1]
        total     = total_llm + total_cb
        print(f"  LLM picks:         {total_llm} ({100*total_llm/max(total,1):.1f}%)")
        print(f"  CB picks:          {total_cb}  ({100*total_cb/max(total,1):.1f}%)")

    if llm_sel and cb_sel:
        print(f"  LLM acc@selected:  {llm_sel[-1]:.4f}")
        print(f"  CB  acc@selected:  {cb_sel[-1]:.4f}")

    # Verdict
    print()
    if final_hybrid is not None and final_llm is not None and final_cold is not None:
        beats_llm  = final_hybrid > final_llm
        beats_cold = final_hybrid > final_cold
        print(f"  Hybrid > LLM alone?      {'YES ✓' if beats_llm  else 'NO ✗'}")
        print(f"  Hybrid > Cold LinUCB?    {'YES ✓' if beats_cold else 'NO ✗'}")

        if beats_llm and beats_cold:
            print("  → HYBRID WINS. Check the learning curve for when it crossed over.")
        elif beats_llm and not beats_cold:
            print("  → Hybrid beats LLM but not cold LinUCB.")
            print("    Cold LinUCB is getting all the data; hybrid's LinUCB is importance-weighted.")
            print("    Possible fix: increase p_min so CB gets more rounds, or reduce lam.")
        elif not beats_llm and beats_cold:
            print("  → Hybrid beats cold LinUCB but not LLM. LLM is strong on this task.")
            print("    The LLM is ceiling-limited. Try a weaker LLM or larger p_min.")
        else:
            print("  → Hybrid loses to both. Diagnose with the acc@selected numbers above.")
            if llm_sel and cb_sel:
                if cb_sel[-1] < llm_sel[-1] * 0.5:
                    print("    CB acc@selected is very low. LinUCB is not learning from IW data.")
                    print("    Consider: lower lam, higher linucb_alpha, or check feature scaling.")
                if data.get('p_llm') and data['p_llm'][-1] >= 0.95:
                    print("    p_llm stayed near 1.0 — CORRAL never shifted to CB.")
                    print("    CB is consistently worse. IW updates may not be helping.")


def plot_curves(results):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({'font.size': 11})
    except ImportError:
        print("matplotlib not installed. Skipping plots.")
        print("Run: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: reward curves
    ax = axes[0]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray']
    ci = 0

    # Find one LLM and one cold LinUCB to plot as shared baselines
    llm_plotted  = False
    cold_plotted = False

    for name, data in results.items():
        batches = data.get('batch', [])
        if not batches:
            continue

        hybrid = data.get('hybrid', [])
        llm    = data.get('llm', [])
        cold   = data.get('linucb_cold', [])

        if hybrid:
            col = colors[ci % len(colors)]
            ax.plot(batches, hybrid, label=f'Hybrid ({name})', color=col, linewidth=2)
            ci += 1

        if llm and not llm_plotted:
            ax.plot(batches, llm, label='LLM only', color='gray',
                    linewidth=1.5, linestyle='--')
            llm_plotted = True

        if cold and not cold_plotted:
            ax.plot(batches, cold, label='Cold LinUCB', color='black',
                    linewidth=1.5, linestyle=':')
            cold_plotted = True

    ax.set_xlabel('Batch')
    ax.set_ylabel('Average Reward')
    ax.set_title('Reward curves')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right plot: p_llm over time (for CORRAL runs)
    ax2 = axes[1]
    ci = 0
    any_pllm = False
    for name, data in results.items():
        batches = data.get('batch', [])
        p_llm   = data.get('p_llm', data.get('p_llm_actual', []))
        if not p_llm:
            continue
        ax2.plot(batches, p_llm, label=name, color=colors[ci % len(colors)], linewidth=2)
        ci += 1
        any_pllm = True

    if any_pllm:
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('p_llm')
        ax2.set_title('LLM probability over time')
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50/50')
    else:
        ax2.text(0.5, 0.5, 'No p_llm data\n(not a CORRAL run)',
                 ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    out = 'comparison_plot.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to: {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='*', default=None,
                        help='Specific JSON files. Default: all results_*.json')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    if args.files:
        results = {}
        for fp in args.files:
            with open(fp) as f:
                data = json.load(f)
            results[os.path.splitext(os.path.basename(fp))[0]] = data
    else:
        results = load_results()

    if not results:
        print("No result files found. Run an experiment first.")
        return

    print(f"\nFound {len(results)} result file(s): {list(results.keys())}")

    for name, data in results.items():
        print_diagnostics(name, data)

    if not args.no_plot:
        plot_curves(results)


if __name__ == '__main__':
    main()
