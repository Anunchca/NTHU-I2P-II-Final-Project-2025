#include <utility>
#include <cstdint>
#include "state.hpp"
#include <algorithm>
#include "minimax.hpp"

#include "tt.hpp"
#include "../games/minichess/state.hpp"


/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax without pruning. Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    uint64_t key = state->hash();
    int alpha_orig = alpha;

    TTEntry* tte = ctx.tt.probe(key);
    if (tte && tte->depth >= depth) {
        if (tte->flag == TT_EXACT) return tte->score;
        else if (tte->flag == TT_LOWER) alpha = std::max(alpha, tte->score);
        else if (tte->flag == TT_UPPER) beta = std::min(beta, tte->score);
        if (alpha >= beta) return tte->score;
    }

    history.push(key);

    if(depth <= 0){
        int score = MiniMax::quiescence(state, alpha, beta, history, ply, ctx, p);
        history.pop(key);
        return score;
    }

    std::partition(state->legal_actions.begin(), state->legal_actions.end(), [&](const Move& a) {
        return state->board.board[1 - state->player][a.second.first][a.second.second] != 0;
    });

    int best_score = M_MAX;
    Move best_move_here;
    bool b_search_pv = true;

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();
        int score;
        if (b_search_pv) {
            int raw = same
                ? MiniMax::eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p)
                : MiniMax::eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
            score = same ? raw : -raw;
        } else {
            int raw = same
                ? MiniMax::eval_ctx(next, depth - 1, alpha, alpha + 1, history, ply + 1, ctx, p)
                : MiniMax::eval_ctx(next, depth - 1, -(alpha + 1), -alpha, history, ply + 1, ctx, p);
            score = same ? raw : -raw;

            if (score > alpha && score < beta) {
                int re_raw = same
                    ? MiniMax::eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p)
                    : MiniMax::eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
                score = same ? re_raw : -re_raw;
            }
        }
        delete next;

        if(score > best_score){
            best_score = score;
            best_move_here = action;
        }

        if (score > alpha) {
            alpha = score;
            b_search_pv = false;
        }

        if (alpha >= beta) {
            break;
        }
    }

    history.pop(key);

    TTFlag flag;
    if (best_score <= alpha_orig) flag = TT_UPPER;
    else if (best_score >= beta) flag = TT_LOWER;
    else flag = TT_EXACT;
    ctx.tt.store(key, depth, best_score, flag, best_move_here);

    return best_score;
}


/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(State *state, int depth, GameHistory& history, SearchContext& ctx) {
    // 1. 全域重置，確保節點計數與 TT 狀態正確
    ctx.nodes = 0;
    ctx.seldepth = 0;

    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()) state->get_legal_actions();

    // 2. [Move Ordering] 強制 TT 的最佳步排在第一位，讓 Alpha-Beta 剪枝飛起來
    uint64_t root_key = state->hash();
    TTEntry* root_tte = ctx.tt.probe(root_key);
    if (root_tte && root_tte->valid) {
        auto it = std::find(state->legal_actions.begin(), state->legal_actions.end(), root_tte->best_move);
        if (it != state->legal_actions.end()) {
            std::rotate(state->legal_actions.begin(), it, it + 1);
        }
    }

    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    int alpha = M_MAX;
    int beta = P_MAX;

    // 3. 開始搜尋
    for(auto& action : state->legal_actions) {
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int raw = same
            ? MiniMax::eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p)
            : MiniMax::eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);

        int score = same ? raw : -raw;
        delete next;

        // 4. 安全中斷與更新
        if(score > best_score) {
            best_score = score;
            result.best_move = action;
            ctx.tt.store(root_key, depth, best_score, TT_EXACT, action);
            if(p.report_partial && ctx.on_root_update) {
               ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }

        if (ctx.nodes > 1500000) ctx.stop = true;
        if (ctx.stop) break;

        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
        move_index++;
    }

    // 5. 保底合法步
    if ((result.best_move.second.first == 0 && result.best_move.second.second == 0) && !state->legal_actions.empty()) {
        result.best_move = state->legal_actions[0];
    }

    result.score = best_score;
    return result;
}

int MiniMax::quiescence(
    State *state,
    int alpha,
    int beta,
    GameHistory &history,
    int ply,
    SearchContext &ctx,
    const MMParams &p
    ) {
    ctx.nodes++;

    int stand_pat = state -> evaluate(p.use_kp_eval, p.use_eval_mobility, &history);

    if (stand_pat >= beta) return stand_pat;
    if(stand_pat >= alpha) alpha = stand_pat;

    if (state -> legal_actions.empty() && state -> game_state == UNKNOWN) {
        state->get_legal_actions();
    }

    for (auto& action : state -> legal_actions) {
        bool is_capture = (state -> board.board[1 - state ->player][action.second.first][action.second.second] != 0);

        if (!is_capture) {
            continue;
        }

        State* next = static_cast<State*>(state -> next_state(action));

        bool same = next->same_player_as_parent();
        int raw = same
            ? MiniMax::quiescence(next, alpha, beta, history, ply + 1, ctx,  p)
            : MiniMax::quiescence(next, -beta, -alpha, history, ply + 1, ctx,  p);

        int score = same ? raw : -raw;
        delete next;

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "false"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "false"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}