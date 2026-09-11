import {
  type CSSProperties,
  type DragEvent as ReactDragEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  type WheelEvent as ReactWheelEvent,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  getTeachingWorkspaceTree,
  selectTeachingWorkspaceDirectory,
  type LocalExecutionDevice,
  type LocalModelProfile,
  type TeachingWorkspaceTree,
  type TeachingWorkspaceTreeEntry,
} from './api';

type EditorNodeType = 'input' | 'workspace' | 'user_input' | 'system_prompt' | 'tool_definition' | 'react_loop' | 'agent' | 'tool' | 'condition' | 'ask_user' | 'final';

export interface EditorNode {
  id: string;
  type: EditorNodeType;
  label: string;
  x: number;
  y: number;
  width?: number;
  height?: number;
  parentId?: string | null;
  config: Record<string, string>;
}

export interface EditorEdge {
  id: string;
  from: string;
  to: string;
  label?: string | null;
}

export interface WorkflowDocument {
  version: '0.1';
  name: string;
  max_steps: number;
  nodes: EditorNode[];
  edges: EditorEdge[];
}

interface PaletteItem {
  type: EditorNodeType;
  label: string;
  description: string;
}

interface MovingNode {
  id: string;
  offsetX: number;
  offsetY: number;
  startX: number;
  startY: number;
  dragged: boolean;
}

interface DemoEvent {
  type: string;
  node: string;
  step: number;
  timestamp: number;
  data: Record<string, unknown>;
}

interface DemoExecutionStats {
  decisions: number;
  toolDecisions: number;
  finalAnswerDecisions: number;
  askUserDecisions: number;
  routeDecisions: number;
  decisionErrors: number;
  toolPlanned: number;
  toolValidated: number;
  toolStarted: number;
  observations: number;
  toolErrors: number;
  policyBlocks: number;
  decisionErrorsByKind: Record<string, number>;
}

type ReactStage = 'context' | 'llm' | 'action' | 'validation' | 'invoke' | 'observation' | 'assemble';

interface ContextGrowthMessage {
  role: string;
  content: unknown;
  name?: string;
  tool_calls?: unknown[];
}

interface ReActPlayback {
  messages: ContextGrowthMessage[];
  initialMessageCount: number;
  activeStages: Set<ReactStage>;
  activeTool: string | null;
  actionLabel: string;
}

interface CanvasAnchor {
  x: number;
  y: number;
}

function workspaceFileSize(bytes?: number): string {
  if (bytes === undefined) return '';
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function WorkspaceTreeView({ entries, depth = 0 }: { entries: TeachingWorkspaceTreeEntry[]; depth?: number }) {
  return (
    <div style={{ display: 'grid', gap: 3, marginLeft: depth ? 12 : 0, borderLeft: depth ? '1px solid #dce7d8' : undefined, paddingLeft: depth ? 8 : 0 }}>
      {entries.map((entry) => (
        <div key={entry.relative_path}>
          <div title={entry.relative_path} style={{ display: 'flex', alignItems: 'baseline', gap: 5, color: entry.kind === 'directory' ? '#356c2a' : entry.readable === false ? '#8b6b45' : '#4d586a', fontSize: 12, lineHeight: 1.45, wordBreak: 'break-all' }}>
            <span aria-hidden="true">{entry.kind === 'directory' ? '▾' : entry.readable === false ? '◌' : '•'}</span>
            <span>{entry.name}</span>
            {entry.kind === 'file' && <span style={{ marginLeft: 'auto', whiteSpace: 'nowrap', color: '#8a96a8', fontSize: 10 }}>{workspaceFileSize(entry.size_bytes)}{entry.readable === false ? ' · 不支持' : ''}</span>}
          </div>
          {entry.kind === 'directory' && entry.children && <WorkspaceTreeView entries={entry.children} depth={depth + 1} />}
        </div>
      ))}
    </div>
  );
}

const decisionReasonLabels: Record<string, string> = {
  connection: '连接', protocol: '协议/JSON', action_validation: '动作校验',
  route: '路由', capability: '能力边界', workflow: '图结构', other: '其他',
};

const STORAGE_KEY = 'agentscratch.workflow.editor.v2';

function persistDocument(document: WorkflowDocument) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(document));
}
const NODE_WIDTH = 188;
const NODE_HEIGHT = 112;

function toolCallCount(action: unknown): number {
  if (!action || typeof action !== 'object') return 0;
  const record = action as Record<string, unknown>;
  if (record.action === 'call_tool') return 1;
  return record.action === 'call_tools' && Array.isArray(record.tool_calls) ? record.tool_calls.length : 0;
}

function accumulateDemoStats(events: DemoEvent[]): DemoExecutionStats {
  return events.reduce<DemoExecutionStats>((stats, event) => {
    if (event.type === 'llm.end') {
      stats.decisions += 1;
      const action = event.data.action as Record<string, unknown> | undefined;
      const actionType = action?.action;
      const callCount = toolCallCount(action);
      stats.toolPlanned += callCount;
      if (callCount > 0) stats.toolDecisions += 1;
      else if (actionType === 'final_answer') stats.finalAnswerDecisions += 1;
      else if (actionType === 'ask_user') stats.askUserDecisions += 1;
      else if (actionType === 'route') stats.routeDecisions += 1;
    } else if (event.type === 'action.validate') {
      stats.toolValidated += toolCallCount(event.data.action);
    } else if (event.type === 'llm.error') {
      stats.decisionErrors += 1;
      const reason = typeof event.data.reason_kind === 'string' ? event.data.reason_kind : 'other';
      stats.decisionErrorsByKind[reason] = (stats.decisionErrorsByKind[reason] || 0) + 1;
    } else if (event.type === 'tool.start') {
      stats.toolStarted += 1;
    } else if (event.type === 'tool.end') {
      stats.observations += 1;
    } else if (event.type === 'tool.error') {
      stats.toolErrors += 1;
    } else if (event.type === 'tool.rejected') {
      stats.policyBlocks += Array.isArray(event.data.calls) ? event.data.calls.length : 1;
    }
    return stats;
  }, { decisions: 0, toolDecisions: 0, finalAnswerDecisions: 0, askUserDecisions: 0, routeDecisions: 0, decisionErrors: 0, toolPlanned: 0, toolValidated: 0, toolStarted: 0, observations: 0, toolErrors: 0, policyBlocks: 0, decisionErrorsByKind: {} });
}

function elapsedLabel(events: DemoEvent[], currentTimeMs?: number): string {
  const started = events.find((event) => event.type === 'run.start')?.timestamp;
  const ended = currentTimeMs !== undefined ? currentTimeMs / 1000 : events.at(-1)?.timestamp;
  if (!started || !ended || ended < started) return '0.00 秒';
  const milliseconds = Math.round((ended - started) * 1000);
  return milliseconds < 1000 ? `${milliseconds} ms` : `${(milliseconds / 1000).toFixed(2)} 秒`;
}

function runStatusPresentation(status?: string): { label: string; background: string; color: string; border: string } {
  switch (status) {
    case 'completed': return { label: '运行完成', background: '#e8f7ee', color: '#176b43', border: '#65ba87' };
    case 'failed': return { label: '运行失败', background: '#fff0ee', color: '#a33a2b', border: '#df8578' };
    case 'paused': return { label: '已暂停', background: '#fff8e6', color: '#8a5a00', border: '#d9aa4a' };
    case 'waiting_user': return { label: '等待用户输入', background: '#fff8e6', color: '#8a5a00', border: '#d9aa4a' };
    case 'running': return { label: '运行中', background: '#eaf4ff', color: '#1e5c99', border: '#78a9dd' };
    default: return { label: '准备运行', background: '#f1f3f6', color: '#4d586a', border: '#cfd6e2' };
  }
}

function renderedNodeWidth(node: EditorNode): number {
  const fallback = node.type === 'input' ? 380 : node.type === 'react_loop' ? 760 : NODE_WIDTH;
  const minimum = node.type === 'input' ? 320 : node.type === 'react_loop' ? 620 : 130;
  const configured = Number(node.width);
  return Number.isFinite(configured) ? Math.min(1600, Math.max(minimum, Math.round(configured))) : fallback;
}

function renderedNodeHeight(node: EditorNode): number {
  const fallback = node.type === 'input' ? 268 : node.type === 'react_loop' ? 490 : NODE_HEIGHT;
  const minimum = node.type === 'input' ? 240 : node.type === 'react_loop' ? 450 : 88;
  const configured = Number(node.height);
  return Number.isFinite(configured) ? Math.min(1200, Math.max(minimum, Math.round(configured))) : fallback;
}

function actionFromEvent(event: DemoEvent | null): Record<string, unknown> | null {
  const action = event?.data.action;
  return action && typeof action === 'object' ? action as Record<string, unknown> : null;
}

function actionToolName(action: Record<string, unknown> | null): string | null {
  if (!action) return null;
  if (typeof action.tool === 'string' && action.tool) return action.tool;
  const calls = action.tool_calls;
  if (Array.isArray(calls) && calls[0] && typeof calls[0] === 'object') {
    const tool = (calls[0] as Record<string, unknown>).tool;
    return typeof tool === 'string' ? tool : null;
  }
  return null;
}

function actionToolNames(action: Record<string, unknown> | null): string[] {
  if (!action) return [];
  if (typeof action.tool === 'string' && action.tool) return [action.tool];
  if (Array.isArray(action.tool_calls)) {
    return action.tool_calls.flatMap((call) => {
      const tool = call && typeof call === 'object' ? (call as Record<string, unknown>).tool : null;
      return typeof tool === 'string' && tool ? [tool] : [];
    });
  }
  return [];
}

function describeModelActionPlan(action: Record<string, unknown> | null): string {
  const tools = actionToolNames(action);
  if (tools.length) return `模型生成行动计划：调用工具 ${tools.join('、')}`;
  switch (action?.action) {
    case 'final_answer': return '模型生成行动计划：直接回答';
    case 'ask_user': return '模型生成行动计划：向用户追问';
    case 'route': return '模型生成行动计划：交给下一个模型';
    default: return '模型已生成行动计划';
  }
}

function describeDemoPhase(event: DemoEvent): string {
  if (event.type === 'llm.end') return describeModelActionPlan(actionFromEvent(event));
  if (event.type === 'tool.start') return event.node ? `正在调用工具：${event.node}` : '正在调用工具';
  if (event.type === 'tool.end') return '工具已返回执行结果';
  if (event.type === 'context.append') {
    if (event.data.source === 'observation') return '正在将工具执行结果写回上下文';
    if (event.data.source === 'assistant_action') return '正在将行动计划写回上下文';
    return '正在将运行反馈写回上下文';
  }
  return ({
    'run.start': '正在初始化运行',
    'context.build': '正在组装上下文',
    'context.request': '上下文已送入模型',
    'llm.start': '模型正在推理',
    'action.validate': '正在校验行动计划',
    'tool.rejected': '工具调用未获允许',
    'llm.error': '模型生成的行动计划未通过校验',
    'agent.waiting_user': '正在等待用户补充信息',
    'agent.finish': '已输出最终答案',
  } as Record<string, string>)[event.type] || '正在处理运行状态';
}

function contextMessageText(message: ContextGrowthMessage): string {
  if (typeof message.content === 'string') return message.content;
  if (message.content !== null && message.content !== undefined) {
    try {
      return typeof message.content === 'object' ? JSON.stringify(message.content) : String(message.content);
    } catch {
      return String(message.content);
    }
  }
  if (message.tool_calls?.length) return JSON.stringify(message.tool_calls);
  return '';
}

function estimatedMessageTokens(message: ContextGrowthMessage): number {
  // A presentation estimate only. Real token counts depend on the selected
  // model tokenizer, which may not be exposed by a local server.
  return Math.max(1, Math.ceil([...contextMessageText(message)].length / 3));
}

function contextMessageLabel(message: ContextGrowthMessage): string {
  if (message.role === 'tool') return `观察 · ${message.name || 'tool'}`;
  if (message.role === 'tool_definition') return `工具定义 · ${message.name || '工具'}`;
  if (message.role === 'assistant') return message.tool_calls?.length ? 'Assistant · tool_calls' : 'Assistant';
  if (message.role === 'system') return 'System';
  return message.role === 'user' ? 'User' : message.role;
}

function replayReActContext(events: DemoEvent[], pendingPhase?: 'context' | 'llm' | null): ReActPlayback {
  const firstSnapshot = events.find((event) => event.type === 'context.build');
  const lastSnapshotIndex = events.reduce((latest, event, index) => event.type === 'context.build' ? index : latest, -1);
  const snapshot = lastSnapshotIndex >= 0 ? events[lastSnapshotIndex] : firstSnapshot;
  const snapshotMessages = Array.isArray(snapshot?.data.messages)
    ? snapshot?.data.messages.filter((message): message is ContextGrowthMessage => Boolean(message) && typeof message === 'object')
    : [];
  const initialMessageCount = Array.isArray(firstSnapshot?.data.messages) ? firstSnapshot.data.messages.length : snapshotMessages.length;
  const requestSnapshot = [...events].reverse().find((event) => event.type === 'context.request');
  // During ``context.build`` the request has not been dispatched yet.  The
  // backend includes the assembled capability definitions on that event, so
  // the yellow tool section is already visible before ``context.request``.
  const toolSnapshot = requestSnapshot || snapshot;
  const requestTools = Array.isArray(toolSnapshot?.data.tools) ? toolSnapshot.data.tools : [];
  const toolDefinitions = requestTools.flatMap((tool) => {
    if (!tool || typeof tool !== 'object') return [];
    const definition = tool as Record<string, unknown>;
    const name = typeof definition.name === 'string' ? definition.name : '工具';
    return [{ role: 'tool_definition', name, content: definition } satisfies ContextGrowthMessage];
  });
  // Tool schemas are part of every model request, but not conversation
  // messages.  Teach their conceptual assembly order as System -> tools ->
  // User, then render all later Observations/actions as growing Context.
  const baseMessages = snapshotMessages.slice(0, initialMessageCount);
  const systemMessages = baseMessages.filter((message) => message.role === 'system');
  const userMessages = baseMessages.filter((message) => message.role === 'user');
  const otherBaseMessages = baseMessages.filter((message) => message.role !== 'system' && message.role !== 'user');
  const messages = [...systemMessages, ...toolDefinitions, ...userMessages, ...otherBaseMessages, ...snapshotMessages.slice(initialMessageCount)];
  const initialMessageCountWithTools = systemMessages.length + toolDefinitions.length + userMessages.length + otherBaseMessages.length;
  if (lastSnapshotIndex >= 0) {
    events.slice(lastSnapshotIndex + 1).forEach((event) => {
      if (event.type !== 'context.append') return;
      const message = event.data.message;
      if (message && typeof message === 'object') messages.push(message as ContextGrowthMessage);
    });
  }

  const latest = events.at(-1) || null;
  const lastDecisionEvent = [...events].reverse().find((event) => event.type === 'llm.end' || event.type === 'action.validate') || null;
  const action = actionFromEvent(lastDecisionEvent);
  const toolFromLatest = latest?.type === 'tool.start' || latest?.type === 'tool.end' || latest?.type === 'tool.error'
    ? latest.node : actionToolName(action);
  const activeStages = new Set<ReactStage>();
  let actionLabel = '等待 LLM 决策';
  if (pendingPhase === 'context') activeStages.add('context');
  else if (pendingPhase === 'llm') {
    activeStages.add('context');
    activeStages.add('llm');
  } else if (latest) {
    if (latest.type === 'run.start' || latest.type === 'context.build') activeStages.add('context');
    else if (latest.type === 'context.request' || latest.type === 'llm.start') {
      activeStages.add('context');
      activeStages.add('llm');
    } else if (latest.type === 'llm.end') {
      activeStages.add('llm');
      activeStages.add('action');
    } else if (latest.type === 'action.validate') {
      activeStages.add('action');
      activeStages.add('validation');
    } else if (latest.type === 'tool.start') {
      activeStages.add('invoke');
    } else if (latest.type === 'tool.end' || latest.type === 'tool.error') {
      activeStages.add('invoke');
      activeStages.add('observation');
    } else if (latest.type === 'context.append') {
      activeStages.add('context');
      // Every append is the write-back/assembly moment.  Assistant actions
      // also retain their decision and validation provenance, but must not
      // make the visible "写回上下文" stage disappear on the final decision.
      activeStages.add('assemble');
      if (latest.data.source === 'assistant_action') {
        activeStages.add('action');
        activeStages.add('validation');
      }
    } else if (latest.type === 'tool.rejected' || latest.type === 'llm.error') {
      activeStages.add('validation');
      activeStages.add('assemble');
    }
  }
  if (action) {
    const type = String(action.action || '');
    actionLabel = type === 'call_tools' ? `工具调用 × ${toolCallCount(action)}`
      : type === 'call_tool' ? `调用 ${actionToolName(action) || '工具'}`
        : type === 'final_answer' ? '输出最终答案'
          : type === 'ask_user' ? '请求用户补充'
            : type === 'route' ? '路由到下一个 LLM' : '模型动作';
  } else if (latest?.type === 'tool.start') actionLabel = `调用 ${latest.node}`;
  else if (latest?.type === 'tool.end') actionLabel = `${latest.node} 返回结果`;
  else if (latest?.type === 'llm.error') actionLabel = '模型输出未通过解析/校验';
  return { messages, initialMessageCount: initialMessageCountWithTools, activeStages, activeTool: toolFromLatest || null, actionLabel };
}

const palette: PaletteItem[] = [
  { type: 'workspace', label: '教学工作区', description: '授权搜索与读取的本机根目录' },
  { type: 'input', label: '上下文模块', description: '装配消息、提示词和工具定义' },
  { type: 'user_input', label: '用户输入模块', description: '可被包含在 Context 中的初始消息' },
  { type: 'system_prompt', label: '系统提示词模块', description: '被包含在上下文中' },
  { type: 'tool_definition', label: '工具定义模块', description: '被包含在上下文中' },
  { type: 'react_loop', label: 'ReAct 循环模块', description: '包含 Context 与 LLM，并限制循环步数' },
  { type: 'agent', label: 'LLM 模块', description: '配置模型推理与可用工具' },
  { type: 'condition', label: '条件分支', description: '按结果进入 true / false 分支' },
  { type: 'ask_user', label: 'Ask user', description: '暂停并等待用户输入' },
  { type: 'final', label: '输出模块', description: '固定输出或流程收束' },
];

const toolPalette: PaletteItem = { type: 'tool', label: '工具模块', description: '独立注册的 LLM 可调用能力' };

const nodeColors: Record<EditorNodeType, { background: string; border: string; label: string }> = {
  workspace: { background: '#f1f6ed', border: '#78a764', label: '#356c2a' },
  input: { background: '#eaf4ff', border: '#5a9ee8', label: '#1e5c99' },
  user_input: { background: '#f0f8ff', border: '#6ca9df', label: '#245f95' },
  system_prompt: { background: '#eef5ff', border: '#759ad8', label: '#315e9e' },
  tool_definition: { background: '#fff5e5', border: '#e1a64e', label: '#8a5a00' },
  react_loop: { background: '#f5edff', border: '#9a6fd1', label: '#68429b' },
  agent: { background: '#eaf8f1', border: '#54ad7a', label: '#18794e' },
  tool: { background: '#fff5e5', border: '#e1a64e', label: '#8a5a00' },
  condition: { background: '#f5edff', border: '#9a6fd1', label: '#68429b' },
  ask_user: { background: '#fff8e6', border: '#d89437', label: '#8a5a00' },
  final: { background: '#e9f8f0', border: '#38a169', label: '#176b43' },
};

let generatedNodeNumber = 0;

function newNode(type: EditorNodeType, x: number, y: number, number: string | number): EditorNode {
  const definitions: Record<EditorNodeType, { label: string; config: Record<string, string> }> = {
    workspace: { label: '教学工作区', config: { root_path: '' } },
    input: { label: '上下文', config: { user_input_module_id: '', context_window_size: '8192' } },
    user_input: { label: '用户输入', config: { text: '计算 12 * 7' } },
    system_prompt: { label: '系统提示词', config: { content: '你是一个清晰、可靠的教学 Agent。工具成功返回 Observation 后，先判断它是否已满足当前子任务；不要仅为重复获得同一结果而用相同参数再次调用工具。若任务明确要求独立校验、不同子任务或时效性结果，可以重复调用，并在 thought 中说明目的。' } },
    tool_definition: { label: '工具定义', config: { tool_ids: '[]' } },
    react_loop: { label: 'ReAct 循环', config: { max_steps: '10', max_decision_failures: '2', max_tool_failures: '2' } },
    // Empty means "use the active local service model". A teaching case must
    // not preserve the old qwen-tiny placeholder after the learner switches
    // to a GGUF in the model manager.
    agent: { label: 'LLM', config: { brain_type: 'test', model: '', device: 'cpu', prompt: '根据上下文自主选择下一步动作。', default_route: '', tool_ids: '[]', tool_protocol: 'json_action', temperature: '0.2', top_p: '0.9', top_k: '40', max_tokens: '512', repeat_penalty: '1.1', max_attempts: '2' } },
    tool: { label: '工具模块', config: { tool: 'calculator', risk: 'read', permission: 'granted', requires_confirmation: 'false', parallel_safe: 'true' } },
    condition: { label: '条件分支', config: { source: 'last_tool_result', operator: 'greater_than', value: '0' } },
    ask_user: { label: 'Ask user', config: { question_mode: 'runtime', question: '' } },
    final: { label: '回答', config: { answer: '' } },
  };
  const definition = definitions[type];
  return {
    id: `${type}-${number}`,
    type,
    label: definition.label,
    x,
    y,
    config: { ...definition.config },
  };
}

function initialDocument(): WorkflowDocument {
  return {
    version: '0.1',
    name: '我的 Agent 工作流',
    max_steps: 10,
    nodes: [
      newNode('user_input', 24, 320, 1),
      { ...newNode('input', 0, 0, 1), parentId: 'react_loop-1', config: { user_input_module_id: 'user_input-1', context_window_size: '8192' } },
      { ...newNode('system_prompt', 24, 118, 1), parentId: 'input-1' },
      { ...newNode('tool_definition', 1018, 68, 1), parentId: 'input-1', config: { tool_ids: '["tool-1"]' } },
      newNode('tool', 1020, 205, 1),
      newNode('react_loop', 235, 86, 1),
      { ...newNode('agent', 0, 0, 1), parentId: 'react_loop-1', config: { ...newNode('agent', 0, 0, 1).config, tool_ids: '["tool-1"]' } },
      newNode('final', 1240, 360, 1),
    ],
    edges: [
      { id: 'edge-user-loop', from: 'user_input-1', to: 'react_loop-1', label: '注入用户消息' },
      { id: 'edge-loop-output', from: 'react_loop-1', to: 'final-1', label: '最终答案' },
    ],
  };
}

function teachingCaseDocument(caseId: 'direct-answer' | 'tool-call' | 'file-summary' | 'file-choice'): WorkflowDocument {
  if (caseId === 'file-summary') {
    return {
      version: '0.1',
      name: '教学案例 3：搜索文件后读取并总结',
      max_steps: 8,
      nodes: [
        { ...newNode('workspace', 24, 560, 3), config: { root_path: '' } },
        { ...newNode('user_input', 24, 372, 3), config: { text: '请在“季度报告”目录中找到“华东销售复盘.md”，阅读后概括其中三项主要结论。' } },
        { ...newNode('system_prompt', 24, 116, 3), parentId: 'input-3', config: { content: '你是可靠的文件总结 Agent。先判断用户是否给出了足以定位文件的信息；不足时用 ask_user 追问，不能猜测目录或 file_id。调用 search_files 时，可省略 directory 或使用“.”搜索授权根目录；只能用成功搜索 Observation 返回的真实 file_id 调用 read_file。读取后，按用户要求总结并结束。' } },
        { ...newNode('input', 0, 0, 3), parentId: 'react_loop-3', config: { user_input_module_id: 'user_input-3', context_window_size: '8192' } },
        { ...newNode('tool_definition', 1018, 68, 3), parentId: 'input-3', config: { tool_ids: '["search-files-3","read-file-3"]' } },
        { ...newNode('react_loop', 255, 86, 3), config: { max_steps: '8', max_decision_failures: '2', max_tool_failures: '2' } },
        { ...newNode('agent', 0, 0, 3), parentId: 'react_loop-3', config: { ...newNode('agent', 0, 0, 3).config, brain_type: 'test', tool_ids: '["search-files-3","read-file-3"]', prompt: '先搜索、再读取、最后总结。每一步只做完成当前子目标所需的行动。' } },
        { ...newNode('tool', 1120, 170, 3), id: 'search-files-3', label: '搜索文件', config: { tool: 'search_files', risk: 'read', permission: 'granted', requires_confirmation: 'false', parallel_safe: 'true' } },
        { ...newNode('tool', 1120, 352, 4), id: 'read-file-3', label: '读取文件', config: { tool: 'read_file', risk: 'read', permission: 'granted', requires_confirmation: 'false', parallel_safe: 'false' } },
        { ...newNode('final', 1540, 310, 3), label: '回答', config: { answer: '' } },
      ],
      edges: [
        { id: 'edge-user-loop-3', from: 'user_input-3', to: 'react_loop-3', label: '注入用户消息' },
        { id: 'edge-loop-output-3', from: 'react_loop-3', to: 'final-3', label: '最终答案' },
      ],
    };
  }
  if (caseId === 'file-choice') {
    return {
      version: '0.1',
      name: '教学案例 4：搜索后追问并总结文件',
      max_steps: 8,
      nodes: [
        { ...newNode('workspace', 24, 584, 4), config: { root_path: '' } },
        { ...newNode('user_input', 24, 378, 4), config: { text: '请先查询工作目录，再让我选择要总结的文件。我暂不指定具体文件名；请根据搜索结果给出最可能的 1-3 个选项。' } },
        { ...newNode('system_prompt', 24, 116, 4), parentId: 'input-4', config: { content: '你是可靠的文件总结 Agent。用户未给出文件名时，先调用 search_files 浏览授权工作目录的文件元数据：使用 directory="."，并省略 file_name。根据成功的搜索 Observation，列出最多 3 个候选，并由任务决定 ask_user 的单选或多选：args.response_schema 必须包含 type:file_selection、source_observation、candidate_file_ids、allow_custom_input:true、selection_mode:single 或 multiple、min_selections、max_selections。用户确认后，只能用该搜索 Observation 中经用户授权的 file_id 调用 read_file；多选时可分别读取每个授权 ID。' } },
        { ...newNode('input', 0, 0, 4), parentId: 'react_loop-4', config: { user_input_module_id: 'user_input-4', context_window_size: '8192' } },
        { ...newNode('tool_definition', 1018, 68, 4), parentId: 'input-4', config: { tool_ids: '["search-files-4","read-file-4"]' } },
        { ...newNode('react_loop', 255, 86, 4), config: { max_steps: '8', max_decision_failures: '2', max_tool_failures: '2' } },
        { ...newNode('agent', 0, 0, 4), parentId: 'react_loop-4', config: { ...newNode('agent', 0, 0, 4).config, brain_type: 'test', tool_ids: '["search-files-4","read-file-4"]', prompt: '先浏览工作目录，再基于搜索 Observation 追问用户。用户选择后读取对应 file_id，最后输出总结。' } },
        { ...newNode('tool', 1120, 152, 5), id: 'search-files-4', label: '搜索文件', config: { tool: 'search_files', risk: 'read', permission: 'granted', requires_confirmation: 'false', parallel_safe: 'true' } },
        { ...newNode('tool', 1120, 330, 6), id: 'read-file-4', label: '读取文件', config: { tool: 'read_file', risk: 'read', permission: 'granted', requires_confirmation: 'false', parallel_safe: 'false' } },
        { ...newNode('ask_user', 1120, 508, 4), label: '询问要总结的文件', config: { question_mode: 'runtime', question: '' } },
        { ...newNode('final', 1540, 330, 4), label: '回答', config: { answer: '' } },
      ],
      edges: [
        { id: 'edge-user-loop-4', from: 'user_input-4', to: 'react_loop-4', label: '注入用户消息' },
        { id: 'edge-loop-output-4', from: 'react_loop-4', to: 'final-4', label: '最终答案' },
      ],
    };
  }
  const toolCase = initialDocument();
  if (caseId === 'tool-call') {
    return { ...toolCase, name: '教学案例 2：调用计算器' };
  }
  const nodes = toolCase.nodes
    .filter((node) => node.type !== 'tool' && node.type !== 'tool_definition')
    .map((node) => node.type === 'user_input'
      ? { ...node, config: { ...node.config, text: '什么是 Agent？' } }
      : node.type === 'agent'
        ? { ...node, config: { ...node.config, tool_ids: '[]' } }
        : node,
    );
  return { ...toolCase, name: '教学案例 1：直接回答', nodes };
}

export function persistWorkflowDocument(document: WorkflowDocument) {
  persistDocument(document);
}

function loadDocument(): WorkflowDocument {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return initialDocument();
    const parsed = JSON.parse(raw) as WorkflowDocument;
    if (!parsed || parsed.version !== '0.1' || !Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
      return initialDocument();
    }
    return normalizeToolReferences({
      ...parsed,
      max_steps: Math.min(60, Math.max(1, parsed.max_steps ?? 10)),
      // Migrate the old default visual label without changing a learner's
      // deliberately renamed custom module.
      nodes: parsed.nodes.map((node) => node.type === 'agent' && node.label === 'Agent' ? { ...node, label: 'LLM' } : node),
    });
  } catch {
    return initialDocument();
  }
}

function compactContent(node: EditorNode, tools: EditorNode[] = []): string {
  if (node.type === 'workspace') return node.config.root_path ? `授权根目录：${node.config.root_path}` : '使用内置教学文件工作区';
  if (node.type === 'input') return '组装并发送下一轮模型上下文';
  if (node.type === 'user_input') return node.config.text || '等待用户输入';
  if (node.type === 'system_prompt') return node.config.content || '未设置提示词';
  if (node.type === 'tool_definition') {
    const names = selectedModuleIds(node).map((id) => tools.find((tool) => tool.id === id)?.config.tool).filter(Boolean);
    return names.length ? `工具 ${names.join('、')} 被定义在此 Context 中` : '尚未定义工具';
  }
  if (node.type === 'react_loop') return '在此循环中持续组装 Context、推理、行动与观察。';
  if (node.type === 'agent') return node.config.model || '使用当前活动模型';
  if (node.type === 'tool') return `${node.config.tool || '未选择工具'} · 独立能力`;
  if (node.type === 'condition') return `${node.config.source || 'last_tool_result'} ${node.config.operator || 'equals'} ${node.config.value || ''}`.trim();
  if (node.type === 'ask_user') return node.config.question_mode === 'runtime'
    ? '问题由 LLM 在运行时根据 Context 与 Observation 生成'
    : node.config.question || '固定问题未设置';
  return node.config.answer || 'LLM 完成后在此显示回答';
}

function workflowNodeTypeLabel(type: EditorNodeType): string {
  if (type === 'tool') return toolPalette.label;
  return palette.find((item) => item.type === type)?.label || type;
}

function selectedModuleIds(node: EditorNode, key = 'tool_ids'): string[] {
  try {
    const value = JSON.parse(node.config[key] || '[]') as unknown;
    return Array.isArray(value) && value.every((item) => typeof item === 'string') ? value : [];
  } catch {
    return [];
  }
}

function normalizeToolReferences(document: WorkflowDocument): WorkflowDocument {
  const tools = document.nodes.filter((node) => node.type === 'tool');
  const validIds = new Set(tools.map((node) => node.id));
  const searchFilesId = tools.find((node) => node.config.tool === 'search_files')?.id;
  const readFileId = tools.find((node) => node.config.tool === 'read_file')?.id;
  return {
    ...document,
    nodes: document.nodes.map((node) => {
      if (node.type === 'final') {
        const legacyPlaceholder = node.config.answer === '{{last_tool_result}}';
        const legacyLabel = node.label === 'Final answer' || node.label === '输出总结';
        return {
          ...node,
          ...(legacyLabel ? { label: '回答' } : {}),
          ...(legacyPlaceholder ? { config: { ...node.config, answer: '' } } : {}),
        };
      }
      if (node.type !== 'tool_definition' && node.type !== 'agent') return node;
      const configured = selectedModuleIds(node).filter((id) => validIds.has(id));
      // A pre-module-editor draft can retain a deleted/randomized module ID.
      // With exactly one available tool there is only one unambiguous repair.
      const repaired = configured.length === 0 && tools.length === 1 ? [tools[0].id] : configured;
      if (readFileId && searchFilesId && repaired.includes(readFileId) && !repaired.includes(searchFilesId)) {
        repaired.push(searchFilesId);
      }
      const needsVisibleToolDefinitionPosition = node.type === 'tool_definition' && node.x === 0 && node.y === 0;
      return {
        ...node,
        ...(needsVisibleToolDefinitionPosition ? { x: 1018, y: 68 } : {}),
        config: { ...node.config, tool_ids: JSON.stringify(repaired) },
      };
    }),
  };
}

const inputStyle: CSSProperties = {
  width: '100%',
  boxSizing: 'border-box',
  padding: 8,
  border: '1px solid #cfd6e2',
  borderRadius: 6,
  font: 'inherit',
};

function ReActStageCard({
  label,
  detail,
  active,
  tone = '#ffffff',
  onSelect,
}: {
  label: string;
  detail?: string;
  active: boolean;
  tone?: string;
  onSelect?: () => void;
}) {
  return (
    <div onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onSelect?.(); }} style={{
      minWidth: 0,
      padding: '7px 8px',
      border: active ? '2px solid #f0a51a' : '1px solid #c8b5df',
      borderRadius: 7,
      background: active ? '#fff6d9' : tone,
      color: '#4b3567',
      boxShadow: active ? '0 0 0 3px #ffcf5b77, 0 3px 10px #c3871633' : undefined,
      transition: 'all 180ms ease',
      textAlign: 'center',
      fontSize: 11,
      fontWeight: 700,
      cursor: onSelect ? 'pointer' : undefined,
    }}>
      <div>{label}</div>
      {detail && <div style={{ marginTop: 3, color: '#6b7280', fontSize: 10, fontWeight: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{detail}</div>}
    </div>
  );
}

function ReActRuntimeDiagram({
  playback,
  demoMode,
  onSelectContext,
  onSelectLLM,
  embeddedModules = [],
  onSelectModule,
  userInputModule,
  systemPromptModules = [],
  toolDefinitionModules = [],
  contextWindowTokens = 8192,
}: {
  playback: ReActPlayback;
  demoMode: boolean;
  onSelectContext?: () => void;
  onSelectLLM?: () => void;
  embeddedModules?: EditorNode[];
  onSelectModule?: (nodeId: string) => void;
  userInputModule?: EditorNode | null;
  systemPromptModules?: EditorNode[];
  toolDefinitionModules?: EditorNode[];
  contextWindowTokens?: number;
}) {
  const [selectedContextMessageIndex, setSelectedContextMessageIndex] = useState<number | null>(null);
  const initialMessages = playback.messages.slice(0, playback.initialMessageCount);
  const grownMessages = playback.messages.slice(playback.initialMessageCount);
  const estimatedTokens = playback.messages.reduce((total, message) => total + estimatedMessageTokens(message), 0);
  const grownTokens = grownMessages.reduce((total, message) => total + estimatedMessageTokens(message), 0);
  const usedRatio = Math.min(1, estimatedTokens / contextWindowTokens);
  const usedPercent = Math.min(999, (estimatedTokens / contextWindowTokens) * 100);
  const selectedContextMessage = selectedContextMessageIndex === null ? null : playback.messages[selectedContextMessageIndex] || null;
  const selectedContextDetail = (() => {
    if (!selectedContextMessage) return '';
    if (typeof selectedContextMessage.content === 'string') return selectedContextMessage.content;
    if (selectedContextMessage.content === null && selectedContextMessage.tool_calls?.length) {
      return JSON.stringify({ tool_calls: selectedContextMessage.tool_calls }, null, 2);
    }
    try {
      return JSON.stringify(selectedContextMessage.content, null, 2);
    } catch {
      return contextMessageText(selectedContextMessage);
    }
  })();
  const isActive = (stage: ReactStage) => playback.activeStages.has(stage);
  const arrow = (active: boolean, label?: string) => (
    <span style={{ color: active ? '#dc8b00' : '#9a6fd1', fontWeight: 800, fontSize: 17, lineHeight: 1, alignSelf: 'center' }} title={label}>→</span>
  );
  const leftArrow = (active: boolean) => (
    <span style={{ color: active ? '#dc8b00' : '#9a6fd1', fontWeight: 800, fontSize: 17, lineHeight: 1, alignSelf: 'center' }}>←</span>
  );
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(245px, 42%) minmax(0, 1fr)', gap: 12, marginTop: 10, userSelect: 'none' }}>
      <div onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onSelectContext?.(); }} style={{ minHeight: 352, padding: 10, border: `2px solid ${isActive('context') ? '#f0a51a' : '#789dd5'}`, borderRadius: 9, background: isActive('context') ? '#fff8df' : '#f6faff', boxShadow: isActive('context') ? '0 0 0 3px #ffcf5b66' : undefined, transition: 'all 180ms ease', cursor: onSelectContext ? 'pointer' : undefined }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6, alignItems: 'baseline' }}>
          <strong style={{ color: '#245f95', fontSize: 12 }}>上下文</strong>
          <span style={{ color: '#5b6475', fontSize: 10 }}>约 {estimatedTokens} / {contextWindowTokens} tokens · {usedPercent.toFixed(1)}%</span>
        </div>
        {!demoMode && (userInputModule || systemPromptModules.length > 0 || toolDefinitionModules.length > 0) && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 5, margin: '5px 0 8px' }}>
            {systemPromptModules.map((module) => <button key={module.id} data-context-reference-id={module.id} onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onSelectModule?.(module.id); }} style={{ padding: '5px 6px', border: '1px solid #759ad8', borderRadius: 5, background: '#eef5ff', color: '#315e9e', fontSize: 10, cursor: onSelectModule ? 'pointer' : 'default', textAlign: 'left' }}>系统提示词 ↗ {module.label}</button>)}
            {toolDefinitionModules.map((module) => <button key={module.id} data-context-reference-id={module.id} onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onSelectModule?.(module.id); }} style={{ padding: '5px 6px', border: '1px solid #e1a64e', borderRadius: 5, background: '#fff5e5', color: '#8a5a00', fontSize: 10, cursor: onSelectModule ? 'pointer' : 'default', textAlign: 'left' }}>工具定义 ↗ {module.label}</button>)}
            {userInputModule && <button data-context-reference-id={userInputModule.id} onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onSelectModule?.(userInputModule.id); }} style={{ padding: '5px 6px', border: '1px solid #6ca9df', borderRadius: 5, background: '#f0f8ff', color: '#245f95', fontSize: 10, cursor: onSelectModule ? 'pointer' : 'default', textAlign: 'left' }}>用户输入 ↗ {userInputModule.label}</button>}
          </div>
        )}
        {!demoMode && embeddedModules.length > 0 && (
          <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', margin: '0 0 8px' }}>
            {embeddedModules.map((module) => (
              <button key={module.id} onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); onSelectModule?.(module.id); }} style={{ padding: '4px 6px', border: '1px solid #9db7de', borderRadius: 5, background: '#ffffff', color: '#315e9e', fontSize: 10, cursor: 'pointer' }}>
                配置 {module.label}
              </button>
            ))}
          </div>
        )}
        <div style={{ position: 'relative', minHeight: 30, height: 30, overflow: 'hidden', borderRadius: 6, background: '#e1e5ea' }}>
          <div style={{ display: 'flex', width: `${usedRatio * 100}%`, minWidth: playback.messages.length ? 2 : 0, height: 30, overflow: 'hidden' }}>
            {playback.messages.map((message, index) => {
              const grown = index >= playback.initialMessageCount;
              const toolDefinition = message.role === 'tool_definition';
              return <div key={`${message.role}-${index}`} title={`${contextMessageLabel(message)}\n${contextMessageText(message)}`} style={{ flex: `${estimatedMessageTokens(message)} 1 0`, minWidth: 1, background: grown ? '#7ed69d' : toolDefinition ? '#f2c14f' : '#73b5ef' }} />;
            })}
          </div>
          {estimatedTokens > contextWindowTokens && <div style={{ position: 'absolute', right: 5, top: 5, padding: '2px 5px', borderRadius: 4, background: '#fff0ee', color: '#a33a2b', fontSize: 9, fontWeight: 700 }}>超过窗口</div>}
        </div>
        <div onPointerDown={(event) => event.stopPropagation()} style={{ marginTop: 8, display: 'grid', gap: 5, height: demoMode ? 278 : 236, overflowY: 'auto', overscrollBehavior: 'contain', touchAction: 'pan-y', paddingRight: 3 }}>
          {initialMessages.map((message, index) => (
            <button key={`initial-${index}`} onClick={(event) => { event.stopPropagation(); setSelectedContextMessageIndex(index); }} style={{ padding: '5px 6px', border: selectedContextMessageIndex === index ? '2px solid #f0a51a' : '1px solid transparent', borderRadius: 5, background: message.role === 'tool_definition' ? '#fff3cf' : '#e8f2ff', color: message.role === 'tool_definition' ? '#805300' : '#315e9e', fontSize: 10, textAlign: 'left', cursor: 'pointer' }}>
              {contextMessageLabel(message)} · {contextMessageText(message).slice(0, 72) || '（结构化工具调用）'}
            </button>
          ))}
          {grownMessages.length > 0 && <div style={{ marginTop: 2, padding: '4px 6px', borderTop: '1px dashed #65ba87', color: '#176b43', fontSize: 10, fontWeight: 700 }}>运行中新增 · {grownMessages.length} 条 / 约 {grownTokens} tokens</div>}
          {grownMessages.map((message, index) => (
            <button key={`grown-${index}`} onClick={(event) => { event.stopPropagation(); setSelectedContextMessageIndex(playback.initialMessageCount + index); }} style={{ padding: '5px 6px', border: selectedContextMessageIndex === playback.initialMessageCount + index ? '2px solid #f0a51a' : '1px solid transparent', borderRadius: 5, background: '#e8f7ee', color: '#176b43', fontSize: 10, textAlign: 'left', cursor: 'pointer' }}>
              + {contextMessageLabel(message)} · {contextMessageText(message).slice(0, 72) || '（结构化工具调用）'}
            </button>
          ))}
        </div>
      </div>

      <div style={{ minWidth: 0, minHeight: 352, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ alignSelf: 'flex-start', width: '100%', boxSizing: 'border-box', padding: 10, border: '1px dashed #b89ade', borderRadius: 9, background: '#fcfaff' }}>
        <div style={{ color: '#68429b', fontSize: 11, fontWeight: 700, marginBottom: 8 }}>本轮执行路径 · {demoMode ? playback.actionLabel : '运行时按真实事件高亮'}</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(72px, 1fr) 18px minmax(72px, 1fr) 18px minmax(72px, 1fr)', alignItems: 'stretch', gap: 3 }}>
          <ReActStageCard label="LLM" detail="生成下一次行动计划" active={isActive('llm')} tone="#eaf8f1" onSelect={onSelectLLM} />
          {arrow(isActive('llm') || isActive('action'))}
          <ReActStageCard label="行动计划" detail="工具 / 回答 / 追问" active={isActive('action')} />
          {arrow(isActive('action') || isActive('validation'))}
          <ReActStageCard label="校验" detail="约束、权限等" active={isActive('validation')} tone="#fff5e5" />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(76px, 1fr) 18px minmax(76px, 1fr) 18px minmax(76px, 1fr)', alignItems: 'stretch', gap: 3 }}>
          <ReActStageCard label="写回上下文" detail="执行结果写回上下文" active={isActive('assemble')} tone="#eef5ff" />
          {leftArrow(isActive('observation') || isActive('assemble'))}
          <ReActStageCard label="观察" detail="工具执行结果" active={isActive('observation')} tone="#fff5e5" />
          {leftArrow(isActive('invoke') || isActive('observation'))}
          <ReActStageCard label="调用" detail={playback.activeTool ? `目标：${playback.activeTool}` : '等待有效工具计划'} active={isActive('invoke')} tone="#fff5e5" />
        </div>
      </div>
      <div style={{ minHeight: 0, flex: 1, display: 'flex', flexDirection: 'column', padding: 10, border: '1px solid #b9cde8', borderRadius: 9, background: '#f7fbff' }}>
        <div style={{ color: '#315e9e', fontSize: 11, fontWeight: 700, marginBottom: 7 }}>上下文详情</div>
        {selectedContextMessage ? (
          <>
            <div style={{ color: selectedContextMessage.role === 'tool_definition' ? '#805300' : '#4d586a', fontSize: 10, fontWeight: 700, marginBottom: 5 }}>{contextMessageLabel(selectedContextMessage)}</div>
            <pre style={{ margin: 0, minHeight: 0, flex: 1, overflowY: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#253a55', fontFamily: 'ui-monospace, Consolas, monospace', fontSize: 10, lineHeight: 1.45 }}>{selectedContextDetail}</pre>
          </>
        ) : <div style={{ color: '#697386', fontSize: 11, lineHeight: 1.45 }}>点击左侧上下文中的任意消息，在此查看完整内容与结构化工具定义。</div>}
      </div>
      </div>
    </div>
  );
}

export default function AgentEditor({
  onRun,
  disabled = false,
  demoMode = false,
  demoEvents = [],
  demoRunStatus,
  demoRunId,
  demoError,
  demoPendingPhase,
  demoOutput = null,
  waitingQuestion = null,
  waitingChoices = [],
  waitingResponseSchema = null,
  onSubmitUserInput,
  onExitDemo,
  localModels = [],
  localDevices = [],
  onActivateLocalModel,
  navigationPanels,
}: {
  onRun?: (workflow: WorkflowDocument) => void | Promise<void>;
  disabled?: boolean;
  demoMode?: boolean;
  demoEvents?: DemoEvent[];
  demoRunStatus?: string;
  demoRunId?: string;
  demoError?: string | null;
  demoPendingPhase?: 'context' | 'llm' | null;
  demoOutput?: string | null;
  waitingQuestion?: string | null;
  waitingChoices?: Array<{ file_id: string; name: string; directory?: string }>;
  waitingResponseSchema?: { type: 'file_selection'; source_observation: string; candidate_file_ids: string[]; allow_custom_input: boolean; selection_mode?: 'single' | 'multiple'; min_selections?: number; max_selections?: number } | null;
  onSubmitUserInput?: (input: string, selectionId?: string, selectionIds?: string[]) => Promise<void>;
  onExitDemo?: () => void;
  localModels?: LocalModelProfile[];
  localDevices?: LocalExecutionDevice[];
  onActivateLocalModel?: (modelId: string, device?: string, contextSize?: number) => Promise<void>;
  navigationPanels?: ReactNode;
}) {
  const [document, setDocument] = useState<WorkflowDocument>(() => loadDocument());
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [connectingFrom, setConnectingFrom] = useState<string | null>(null);
  const [status, setStatus] = useState('草稿已自动保存到本地浏览器');
  const documentRef = useRef(document);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const movingRef = useRef<MovingNode | null>(null);
  const suppressNodeClickRef = useRef<string | null>(null);
  const panningRef = useRef<{ startX: number; startY: number; originX: number; originY: number } | null>(null);
  const [canvasViewportWidth, setCanvasViewportWidth] = useState(0);
  // A numeric cursor is an exclusive event cursor for historical playback;
  // null means follow the latest event while the workflow is running.
  const [demoCursor, setDemoCursor] = useState<number | null>(null);
  const [demoPlaying, setDemoPlaying] = useState(false);
  const [demoSpeed, setDemoSpeed] = useState(1);
  const [playbackPanelCollapsed, setPlaybackPanelCollapsed] = useState(false);
  const [modelSwitching, setModelSwitching] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [canvasZoom, setCanvasZoom] = useState(1);
  const [canvasPan, setCanvasPan] = useState({ x: 0, y: 0 });
  const [canvasMenu, setCanvasMenu] = useState<{ x: number; y: number } | null>(null);
  const [contextReferenceAnchors, setContextReferenceAnchors] = useState<Record<string, CanvasAnchor>>({});
  const [navigationOpen, setNavigationOpen] = useState(false);
  const [workspaceTree, setWorkspaceTree] = useState<TeachingWorkspaceTree | null>(null);
  const [workspaceLoading, setWorkspaceLoading] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [runtimeClockMs, setRuntimeClockMs] = useState(() => Date.now());
  const [waitingInput, setWaitingInput] = useState('');
  const [waitingSelectionIds, setWaitingSelectionIds] = useState<string[]>([]);
  const [waitingSubmitError, setWaitingSubmitError] = useState<string | null>(null);
  const [waitingSubmitting, setWaitingSubmitting] = useState(false);

  useEffect(() => {
    documentRef.current = document;
    persistDocument(document);
  }, [document]);

  useEffect(() => {
    // `pagehide` also covers a quick F5 immediately after dropping a module.
    // The latest drag position is kept in the ref synchronously below.
    const persistBeforeLeave = () => persistDocument(documentRef.current);
    window.addEventListener('pagehide', persistBeforeLeave);
    return () => window.removeEventListener('pagehide', persistBeforeLeave);
  }, []);

  useEffect(() => {
    function move(event: PointerEvent) {
      const panning = panningRef.current;
      if (panning) {
        setCanvasPan({ x: panning.originX + event.clientX - panning.startX, y: panning.originY + event.clientY - panning.startY });
        return;
      }
      const moving = movingRef.current;
      const canvas = canvasRef.current;
      if (!moving || !canvas) return;
      if (Math.hypot(event.clientX - moving.startX, event.clientY - moving.startY) >= 3) {
        moving.dragged = true;
      }
      const rect = canvas.getBoundingClientRect();
      setDocument((current) => {
        const next = {
          ...current,
          nodes: current.nodes.map((node) =>
            node.id === moving.id
              ? {
                  ...node,
                  x: Math.max(8, (event.clientX - rect.left) / canvasZoom - moving.offsetX),
                  y: Math.max(8, (event.clientY - rect.top) / canvasZoom - moving.offsetY),
                }
              : node,
          ),
        };
        // Persist the exact pointer position in the same event.  Relying only
        // on the post-render effect could lose the last movement on a fast F5.
        documentRef.current = next;
        persistDocument(next);
        return next;
      });
    }
    function stop() {
      if (movingRef.current?.dragged) suppressNodeClickRef.current = movingRef.current.id;
      movingRef.current = null;
      panningRef.current = null;
    }
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop);
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
    };
  }, [canvasZoom]);

  useEffect(() => {
    if (!demoMode) {
      setDemoCursor(null);
      setDemoPlaying(false);
      setPlaybackPanelCollapsed(false);
      setWaitingInput('');
      setWaitingSubmitError(null);
    }
  }, [demoMode]);

  useEffect(() => {
    if (demoRunStatus === 'waiting_user') {
      setWaitingInput('');
      setWaitingSubmitError(null);
    }
  }, [demoRunStatus, waitingQuestion]);

  useEffect(() => {
    if (!demoMode || demoRunStatus !== 'running') return;
    setRuntimeClockMs(Date.now());
    const timer = window.setInterval(() => setRuntimeClockMs(Date.now()), 200);
    return () => window.clearInterval(timer);
  }, [demoMode, demoRunStatus]);

  useEffect(() => {
    const viewport = canvasRef.current?.parentElement?.parentElement;
    if (!viewport) return;
    const update = () => setCanvasViewportWidth(viewport.clientWidth);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(viewport);
    return () => observer.disconnect();
  }, []);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const canvasRect = canvas.getBoundingClientRect();
    const next: Record<string, CanvasAnchor> = {};
    canvas.querySelectorAll<HTMLElement>('[data-context-reference-id]').forEach((element) => {
      const id = element.dataset.contextReferenceId;
      if (!id) return;
      const rect = element.getBoundingClientRect();
      const referenceModule = document.nodes.find((node) => node.id === id);
      const pointsToRightSide = referenceModule?.type === 'tool_definition';
      next[id] = {
        // System/User modules enter their chips from the left; the tool
        // definition lives to the right of Context and points to that chip's
        // right middle edge.
        x: ((pointsToRightSide ? rect.right - 3 : rect.left + 3) - canvasRect.left) / canvasZoom,
        y: (rect.top - canvasRect.top + rect.height / 2) / canvasZoom,
      };
    });
    setContextReferenceAnchors((current) => JSON.stringify(current) === JSON.stringify(next) ? current : next);
  }, [document, canvasZoom, demoMode, demoEvents.length, demoCursor]);

  const selectedNode = document.nodes.find((node) => node.id === selectedNodeId) || null;
  const nodeMap = useMemo(() => new Map(document.nodes.map((node) => [node.id, node])), [document.nodes]);
  const selectedOutgoing = selectedNode ? document.edges.filter((edge) => edge.from === selectedNode.id) : [];
  const toolModules = document.nodes.filter((node) => node.type === 'tool');
  const contextModules = document.nodes.filter((node) => node.type === 'input');
  const toolDefinitionModules = document.nodes.filter((node) => node.type === 'tool_definition');
  const contextToolIds = new Set(toolDefinitionModules.flatMap((node) => selectedModuleIds(node)));
  const hasExplicitToolDefinitions = toolDefinitionModules.length > 0;
  const displayedDemoEvents = demoCursor === null ? demoEvents : demoEvents.slice(0, demoCursor);
  const latestDemoEvent = displayedDemoEvents.length ? displayedDemoEvents[displayedDemoEvents.length - 1] : null;
  const visibleFinishEvent = [...displayedDemoEvents].reverse().find((event) => event.type === 'agent.finish');
  const visibleFinalOutput = typeof visibleFinishEvent?.data.output === 'string'
    ? visibleFinishEvent.data.output
    : visibleFinishEvent && demoOutput ? demoOutput : null;
  const demoStats = accumulateDemoStats(displayedDemoEvents);
  const reactPlayback = replayReActContext(displayedDemoEvents, demoPendingPhase);
  const totalElapsed = elapsedLabel(demoEvents);
  const liveElapsed = elapsedLabel(demoEvents, demoRunStatus === 'running' ? runtimeClockMs : undefined);
  const statusPresentation = runStatusPresentation(demoRunStatus);
  const terminalDemo = demoRunStatus === 'completed' || demoRunStatus === 'failed';
  const loopModule = document.nodes.find((node) => node.type === 'react_loop');
  const loopContextModule = loopModule ? document.nodes.find((node) => node.type === 'input' && node.parentId === loopModule.id) || null : null;
  const configuredUserInputId = loopContextModule?.config.user_input_module_id || '';
  const contextUserInputModule = document.nodes.find((node) => node.type === 'user_input' && node.id === configuredUserInputId)
    || document.nodes.find((node) => node.type === 'user_input' && node.parentId === loopContextModule?.id)
    || document.nodes.find((node) => node.type === 'user_input' && !node.parentId)
    || null;
  const contextSystemPromptModules = loopContextModule
    ? document.nodes.filter((node) => node.type === 'system_prompt' && node.parentId === loopContextModule.id)
    : [];
  const contextToolDefinitionModules = loopContextModule
    ? document.nodes.filter((node) => node.type === 'tool_definition' && node.parentId === loopContextModule.id)
    : [];
  const contextWindowTokens = Math.min(131072, Math.max(256, Number(loopContextModule?.config.context_window_size) || 8192));
  const decisionFailureLimit = loopModule?.config.max_decision_failures || '2';
  const toolFailureLimit = loopModule?.config.max_tool_failures || '2';
  const visibleFailureReason = latestDemoEvent?.type === 'agent.error'
    ? String(latestDemoEvent.data.error || demoError || '')
    : demoRunStatus === 'failed' ? demoError || '' : '';
  const waitingEvent = [...displayedDemoEvents].reverse().find((event) => event.type === 'agent.waiting_user');
  const visibleWaitingQuestion = waitingQuestion || (typeof waitingEvent?.data.question === 'string' ? waitingEvent.data.question : '请补充 Agent 所需的信息。');
  const allowsCustomWaitingInput = waitingResponseSchema?.type !== 'file_selection' || waitingResponseSchema.allow_custom_input;
  const multipleFileSelection = waitingResponseSchema?.type === 'file_selection' && waitingResponseSchema.selection_mode === 'multiple';
  const waitingMinSelections = waitingResponseSchema?.min_selections ?? 1;
  const waitingMaxSelections = waitingResponseSchema?.max_selections ?? (multipleFileSelection ? waitingChoices.length : 1);
  const activeWaitingSelectionIds = waitingSelectionIds.filter((fileId) => waitingChoices.some((choice) => choice.file_id === fileId));
  const contextAssemblyModuleIds = [
    contextUserInputModule?.id,
    ...contextSystemPromptModules.map((node) => node.id),
    ...contextToolDefinitionModules.map((node) => node.id),
  ].filter((id): id is string => Boolean(id));
  const demoFocusIds = (() => {
    if (demoPendingPhase === 'llm') return [document.nodes.find((node) => node.type === 'agent')?.id].filter((id): id is string => Boolean(id));
    if (demoPendingPhase === 'context') return [loopContextModule?.id, ...contextAssemblyModuleIds].filter((id): id is string => Boolean(id));
    if (!latestDemoEvent) return [];
    if (latestDemoEvent.type === 'run.start') return contextAssemblyModuleIds;
    if (latestDemoEvent.type === 'context.build') return [loopContextModule?.id, ...contextAssemblyModuleIds].filter((id): id is string => Boolean(id));
    if (latestDemoEvent.type === 'context.request' || latestDemoEvent.type === 'context.append' || latestDemoEvent.type === 'tool.end') return [loopContextModule?.id].filter((id): id is string => Boolean(id));
    if (latestDemoEvent.type === 'llm.start') return [document.nodes.find((node) => node.type === 'agent')?.id].filter((id): id is string => Boolean(id));
    if (latestDemoEvent.type === 'llm.end' || latestDemoEvent.type === 'action.validate') return [document.nodes.find((node) => node.type === 'agent')?.id].filter((id): id is string => Boolean(id));
    if (latestDemoEvent.type === 'tool.start') return [document.nodes.find((node) => node.type === 'tool' && node.config.tool === latestDemoEvent.node)?.id].filter((id): id is string => Boolean(id));
    if (latestDemoEvent.type === 'agent.waiting_user') return [document.nodes.find((node) => node.type === 'ask_user')?.id || document.nodes.find((node) => node.type === 'agent')?.id].filter((id): id is string => Boolean(id));
    if (latestDemoEvent.type === 'agent.finish') return [document.nodes.find((node) => node.type === 'final')?.id].filter((id): id is string => Boolean(id));
    return [];
  })();
  const demoPhase = demoPendingPhase === 'llm'
    ? '模型正在推理'
    : demoPendingPhase === 'context'
      ? '正在组装并发送上下文'
      : latestDemoEvent
    ? describeDemoPhase(latestDemoEvent)
    : '等待运行开始';
  const canvasVisibleNodes = document.nodes.filter((node) => !node.parentId || node.type === 'system_prompt' || node.type === 'tool_definition');
  const canvasWidth = Math.max(1000, canvasViewportWidth, ...canvasVisibleNodes.map((node) => node.x + renderedNodeWidth(node) + 40));
  const canvasHeight = Math.max(640, ...canvasVisibleNodes.map((node) => node.y + renderedNodeHeight(node) + 40));

  async function refreshWorkspaceTree(rootPath = selectedNode?.config.root_path || '') {
    if (!rootPath) {
      setWorkspaceTree(null);
      setWorkspaceError(null);
      return;
    }
    setWorkspaceLoading(true);
    setWorkspaceError(null);
    try {
      setWorkspaceTree(await getTeachingWorkspaceTree(rootPath));
    } catch (reason) {
      setWorkspaceTree(null);
      setWorkspaceError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setWorkspaceLoading(false);
    }
  }

  async function chooseWorkspaceDirectory() {
    setWorkspaceLoading(true);
    setWorkspaceError(null);
    try {
      const selection = await selectTeachingWorkspaceDirectory();
      if (!selection.selected) {
        setStatus('未选择文件夹，教学工作区保持不变');
        return;
      }
      updateConfig('root_path', selection.root_path);
      setWorkspaceTree(selection.tree);
      setStatus(`教学工作区已授权：${selection.root_path}`);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setWorkspaceError(message);
      setStatus(`选择教学工作区失败：${message}`);
    } finally {
      setWorkspaceLoading(false);
    }
  }

  useEffect(() => {
    if (selectedNode?.type !== 'workspace') return;
    void refreshWorkspaceTree(selectedNode.config.root_path || '');
  }, [selectedNode?.id, selectedNode?.type, selectedNode?.config.root_path]);

  function moveDemoCursor(nextCursor: number, stopPlayback = true) {
    const cursor = Math.max(0, Math.min(demoEvents.length, nextCursor));
    setDemoCursor(cursor);
    if (stopPlayback) setDemoPlaying(false);
  }

  async function activateModel(modelId: string, device = selectedNode?.config.device || 'cpu') {
    if (!onActivateLocalModel) {
      updateConfig('model', modelId);
      return;
    }
    setModelSwitching(true);
    try {
      setStatus(`正在加载 ${modelId}…`);
      await onActivateLocalModel(modelId, device, contextWindowTokens);
      updateConfig('model', modelId);
      setStatus(`本地模型已切换为 ${modelId}`);
    } catch (reason) {
      setStatus(`模型切换失败：${reason instanceof Error ? reason.message : String(reason)}`);
    } finally {
      setModelSwitching(false);
    }
  }

  useEffect(() => {
    if (!demoPlaying || demoCursor === null) return undefined;
    if (demoCursor >= demoEvents.length) {
      setDemoPlaying(false);
      return undefined;
    }
    const timer = window.setTimeout(
      () => moveDemoCursor(demoCursor + 1, false),
      Math.round(800 / demoSpeed),
    );
    return () => window.clearTimeout(timer);
  }, [demoCursor, demoEvents.length, demoPlaying, demoSpeed]);

  function addNode(type: EditorNodeType, x = 48, y = 48) {
    generatedNodeNumber += 1;
    const parent = (type === 'system_prompt' || type === 'tool_definition')
      ? (selectedNode?.type === 'input' ? selectedNode.id : contextModules[0]?.id)
      : type === 'agent' && selectedNode?.type === 'react_loop' ? selectedNode.id : undefined;
    const node = { ...newNode(type, x, y, `${Date.now()}-${generatedNodeNumber}`), ...(parent ? { parentId: parent } : {}) };
    setDocument((current) => ({ ...current, nodes: [...current.nodes, node] }));
    setSelectedNodeId(node.id);
    setStatus(`${workflowNodeTypeLabel(type)} 已添加`);
  }

  function handlePaletteDragStart(event: ReactDragEvent<HTMLButtonElement>, type: EditorNodeType) {
    event.dataTransfer.setData('application/x-agentscratch-node', type);
    event.dataTransfer.effectAllowed = 'copy';
  }

  function handleCanvasDrop(event: ReactDragEvent<HTMLDivElement>) {
    event.preventDefault();
    const type = event.dataTransfer.getData('application/x-agentscratch-node') as EditorNodeType;
    if (!palette.some((item) => item.type === type) || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    addNode(type, Math.max(8, (event.clientX - rect.left) / canvasZoom - NODE_WIDTH / 2), Math.max(8, (event.clientY - rect.top) / canvasZoom - 40));
  }

  function handleCanvasWheel(event: ReactWheelEvent<HTMLDivElement>) {
    const target = event.target as HTMLElement;
    const overModule = Boolean(target.closest('[data-workflow-module]'));
    // The empty grid is a camera surface.  Module content keeps its native
    // scrolling behavior unless the learner explicitly requests zoom with Ctrl.
    if (overModule && !event.ctrlKey) return;
    if (!overModule && event.target !== event.currentTarget) return;
    event.preventDefault();
    setCanvasZoom((current) => {
      const delta = event.deltaY < 0 ? 0.1 : -0.1;
      return Math.min(1.8, Math.max(0.45, Number((current + delta).toFixed(2))));
    });
  }

  function startCanvasPan(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || event.target !== event.currentTarget) return;
    panningRef.current = { startX: event.clientX, startY: event.clientY, originX: canvasPan.x, originY: canvasPan.y };
    setSelectedNodeId(null);
    setCanvasMenu(null);
  }

  function centerCanvas() {
    const viewport = canvasRef.current?.parentElement?.parentElement;
    if (!viewport || canvasVisibleNodes.length === 0) return;
    const left = Math.min(...canvasVisibleNodes.map((node) => node.x));
    const right = Math.max(...canvasVisibleNodes.map((node) => node.x + renderedNodeWidth(node)));
    const top = Math.min(...canvasVisibleNodes.map((node) => node.y));
    const bottom = Math.max(...canvasVisibleNodes.map((node) => node.y + renderedNodeHeight(node)));
    setCanvasPan({
      x: viewport.clientWidth / 2 - ((left + right) / 2) * canvasZoom,
      y: viewport.clientHeight / 2 - ((top + bottom) / 2) * canvasZoom,
    });
    setCanvasMenu(null);
  }

  function handleContainerDrop(event: ReactDragEvent<HTMLDivElement>, parent: EditorNode) {
    event.preventDefault();
    event.stopPropagation();
    const type = event.dataTransfer.getData('application/x-agentscratch-node') as EditorNodeType;
    const allowed = parent.type === 'input'
      ? new Set<EditorNodeType>(['user_input', 'system_prompt', 'tool_definition'])
      : new Set<EditorNodeType>(['input', 'agent']);
    if (!allowed.has(type)) {
        setStatus(parent.type === 'input' ? 'Context 只能包含用户输入、系统提示词或工具定义。' : 'ReAct 循环只能包含 Context 或 LLM 模块。');
      return;
    }
    generatedNodeNumber += 1;
    const node = { ...newNode(type, 0, 0, `${Date.now()}-${generatedNodeNumber}`), parentId: parent.id };
    setDocument((current) => ({ ...current, nodes: [...current.nodes, node] }));
    setSelectedNodeId(node.id);
    setStatus(`${workflowNodeTypeLabel(type)} 已放入 ${parent.label}`);
  }

  function startMoving(event: ReactPointerEvent<HTMLDivElement>, node: EditorNode) {
    if (demoMode) return;
    if ((event.target as HTMLElement).dataset.handle) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    movingRef.current = {
      id: node.id,
      offsetX: (event.clientX - rect.left) / canvasZoom - node.x,
      offsetY: (event.clientY - rect.top) / canvasZoom - node.y,
      startX: event.clientX,
      startY: event.clientY,
      dragged: false,
    };
    setConnectingFrom(null);
  }

  function startConnection(event: ReactPointerEvent<HTMLButtonElement>, nodeId: string) {
    if (demoMode) return;
    event.stopPropagation();
    setSelectedNodeId(nodeId);
    setConnectingFrom(nodeId);
  }

  function finishConnection(event: { stopPropagation: () => void }, targetId: string) {
    if (demoMode) return;
    event.stopPropagation();
    connectTo(targetId);
  }

  function connectTo(targetId: string) {
    if (!connectingFrom || connectingFrom === targetId) {
      setConnectingFrom(null);
      return;
    }
    const edgeId = `edge-${connectingFrom}-${targetId}`;
    setDocument((current) => {
      if (current.edges.some((edge) => edge.from === connectingFrom && edge.to === targetId)) return current;
      const source = current.nodes.find((node) => node.id === connectingFrom);
      const target = current.nodes.find((node) => node.id === targetId);
      const procedural = new Set<EditorNodeType>(['user_input', 'input', 'react_loop', 'agent', 'condition', 'ask_user', 'final']);
      if (!source || !target || !procedural.has(source.type) || !procedural.has(target.type) || source.type === 'final') {
        setStatus('工具、提示词和工具定义是模块关联，不使用流程连线。');
        return current;
      }
      if (source.type === 'agent' && target.type !== 'agent') {
        setStatus('LLM 的工具由“可用工具”配置；实线只用于路由到下一位 LLM。');
        return current;
      }
      const existing = current.edges.filter((edge) => edge.from === connectingFrom);
      if (source?.type === 'condition' && existing.length >= 2) {
        setStatus('条件节点只能有 true 和 false 两条出线');
        return current;
      }
      const labels = new Set(existing.map((edge) => edge.label));
      const label = source?.type === 'condition' ? (labels.has('true') ? 'false' : 'true') : undefined;
      return { ...current, edges: [...current.edges, { id: edgeId, from: connectingFrom, to: targetId, label }] };
    });
    setConnectingFrom(null);
    setStatus('连接已添加');
  }

  function selectNode(nodeId: string) {
    if (connectingFrom && connectingFrom !== nodeId) {
      connectTo(nodeId);
      return;
    }
    setSelectedNodeId(nodeId);
  }

  function deleteSelected() {
    if (!selectedNode) return;
    setDocument((current) => ({
      ...current,
      nodes: current.nodes.filter((node) => node.id !== selectedNode.id),
      edges: current.edges.filter((edge) => edge.from !== selectedNode.id && edge.to !== selectedNode.id),
    }));
    setSelectedNodeId('');
    setStatus('节点及其连线已删除');
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement;
      if (target.matches('input, textarea, select')) return;
      if (event.key === 'Escape') {
        setConnectingFrom(null);
        return;
      }
      if ((event.key === 'Delete' || event.key === 'Backspace') && selectedNodeId) {
        event.preventDefault();
        deleteSelected();
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  });

  function updateSelected(patch: Partial<EditorNode>) {
    if (!selectedNodeId) return;
    setDocument((current) => ({
      ...current,
      nodes: current.nodes.map((node) => (node.id === selectedNodeId ? { ...node, ...patch } : node)),
    }));
  }

  function updateConfig(key: string, value: string) {
    if (!selectedNodeId) return;
    // Model activation is asynchronous.  Merge against the latest document
    // node so an earlier snapshot cannot overwrite a newly selected device.
    setDocument((current) => ({
      ...current,
      nodes: current.nodes.map((node) => node.id === selectedNodeId
        ? { ...node, config: { ...node.config, [key]: value } }
        : node),
    }));
  }

  function toggleModuleId(key: string, moduleId: string, checked: boolean) {
    if (!selectedNode) return;
    setDocument((current) => {
      const selectedTool = current.nodes.find((node) => node.id === moduleId);
      const searchTool = current.nodes.find((node) => node.type === 'tool' && node.config.tool === 'search_files');
      const readTool = current.nodes.find((node) => node.type === 'tool' && node.config.tool === 'read_file');
      return {
        ...current,
        nodes: current.nodes.map((node) => {
          if (node.id !== selectedNode.id) return node;
          const ids = new Set(selectedModuleIds(node, key));
          if (checked) ids.add(moduleId);
          else ids.delete(moduleId);
          // read_file is deliberately not an independent capability: it only
          // accepts opaque IDs emitted by search_files.  Keep the dependency
          // visible in both Context definitions and LLM capability lists.
          if (selectedTool?.config.tool === 'read_file' && checked && searchTool) ids.add(searchTool.id);
          if (selectedTool?.config.tool === 'search_files' && !checked && readTool) ids.delete(readTool.id);
          return { ...node, config: { ...node.config, [key]: JSON.stringify([...ids]) } };
        }),
      };
    });
  }

  function deleteEdge(edgeId: string) {
    setDocument((current) => ({ ...current, edges: current.edges.filter((edge) => edge.id !== edgeId) }));
    setStatus('连线已删除');
  }

  function updateEdgeLabel(edgeId: string, label: string) {
    setDocument((current) => ({
      ...current,
      edges: current.edges.map((edge) => (edge.id === edgeId ? { ...edge, label } : edge)),
    }));
  }

  function autoLayout() {
    setDocument((current) => ({
      ...current,
      nodes: current.nodes.map((node, index) => node.parentId ? node : {
        ...node,
        x: 28 + (index % 4) * 242,
        y: 80 + Math.floor(index / 4) * 170,
      }),
    }));
    setStatus('已自动排列节点');
  }

  function resetDocument() {
    if (!window.confirm('确定要清空当前编排并恢复示例流程吗？')) return;
    const fresh = initialDocument();
    setDocument(fresh);
    setSelectedNodeId(null);
    setStatus('已恢复示例流程');
  }

  function loadTeachingCase(caseId: 'direct-answer' | 'tool-call' | 'file-summary' | 'file-choice') {
    const next = teachingCaseDocument(caseId);
    setDocument(next);
    setSelectedNodeId(null);
    setStatus(caseId === 'direct-answer'
      ? '已载入案例 1：一问一答，不调用工具'
      : caseId === 'tool-call'
        ? '已载入案例 2：调用 calculator 后观察结果'
        : caseId === 'file-summary'
          ? '已载入案例 3：先搜索文件，再读取并总结'
          : '已载入案例 4：先搜索候选文件，再 ask_user 续跑');
  }

  function exportDocument() {
    const blob = new Blob([JSON.stringify(document, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = window.document.createElement('a');
    anchor.href = url;
    anchor.download = `${document.name || 'agentscratch-workflow'}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    setStatus('工作流 JSON 已导出');
  }

  async function submitWaitingInput(inputOverride?: string, selectionId?: string, selectionIds?: string[]) {
    const input = (inputOverride ?? waitingInput).trim();
    if ((!input && !selectionId && !selectionIds?.length) || !onSubmitUserInput) return;
    setWaitingSubmitting(true);
    setWaitingSubmitError(null);
    try {
      await onSubmitUserInput(input, selectionId, selectionIds);
      setWaitingInput('');
      setWaitingSelectionIds([]);
    } catch (reason) {
      setWaitingSubmitError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setWaitingSubmitting(false);
    }
  }

  function runCurrentWorkflow() {
    const normalized = normalizeToolReferences(document);
    setDocument(normalized);
    setSelectedNodeId(null);
    return onRun?.(normalized);
  }

  return (
    <section style={{ position: 'fixed', inset: 0, overflow: 'hidden', background: '#fbfcff' }}>
      <nav
        aria-label="AgentScratch 工作台导航"
        onMouseEnter={() => setNavigationOpen(true)}
        onMouseLeave={() => setNavigationOpen(false)}
        style={{ position: 'fixed', zIndex: 40, top: 0, left: 0, right: 0, maxHeight: 'calc(100vh - 16px)', overflowY: 'auto', boxSizing: 'border-box', padding: '14px clamp(16px, 2.5vw, 42px) 20px', borderBottom: '1px solid #cfd8e6', borderRadius: '0 0 14px 14px', background: '#ffffff', boxShadow: '0 9px 30px #17203324', transform: navigationOpen ? 'translateY(0)' : 'translateY(calc(-100% + 16px))', transition: 'transform 220ms ease' }}
      >
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 280px' }}>
          <h1 style={{ fontSize: 20, margin: 0 }}>AgentScratch</h1>
          <p style={{ color: '#5b6475', margin: '6px 0 0' }}>
            {demoMode ? '运行与回放由真实事件时间线驱动。' : '全屏编排画布 · 空白区滚轮缩放，模块内按住 Ctrl 再滚轮缩放。'}
          </p>
        </div>
        <input
          aria-label="工作流名称"
          value={document.name}
          onChange={(event) => setDocument((current) => ({ ...current, name: event.target.value }))}
          style={{ ...inputStyle, width: 220 }}
        />
        <label style={{ color: '#4d586a', fontSize: 12 }}>
          最大步数
          <input
            aria-label="最大执行步数"
            type="number"
            min="1"
            max="60"
            value={document.max_steps}
            disabled={demoMode}
            onChange={(event) => {
              const maxSteps = Math.min(60, Math.max(1, Number(event.target.value) || 1));
              setDocument((current) => ({
                ...current,
                max_steps: maxSteps,
                nodes: current.nodes.map((node) => node.type === 'react_loop' ? { ...node, config: { ...node.config, max_steps: String(maxSteps) } } : node),
              }));
            }}
            style={{ ...inputStyle, width: 64, marginLeft: 6 }}
          />
        </label>
        <label style={{ color: '#4d586a', fontSize: 12 }}>
          最大决策错误
          <input
            aria-label="最大决策错误次数"
            type="number"
            min="0"
            max="60"
            value={decisionFailureLimit}
            disabled={demoMode}
            onChange={(event) => {
              const value = String(Math.min(60, Math.max(0, Number(event.target.value) || 0)));
              setDocument((current) => ({ ...current, nodes: current.nodes.map((node) => node.type === 'react_loop' ? { ...node, config: { ...node.config, max_decision_failures: value } } : node) }));
            }}
            style={{ ...inputStyle, width: 58, marginLeft: 6 }}
          />
        </label>
        <label style={{ color: '#4d586a', fontSize: 12 }}>
          最大工具失败
          <input
            aria-label="最大工具执行失败次数"
            type="number"
            min="0"
            max="10"
            value={toolFailureLimit}
            disabled={demoMode}
            onChange={(event) => {
              const value = String(Math.min(10, Math.max(0, Number(event.target.value) || 0)));
              setDocument((current) => ({ ...current, nodes: current.nodes.map((node) => node.type === 'react_loop' ? { ...node, config: { ...node.config, max_tool_failures: value } } : node) }));
            }}
            style={{ ...inputStyle, width: 58, marginLeft: 6 }}
          />
        </label>
        <button onClick={() => setPaletteOpen(true)} disabled={demoMode}>打开模块库</button>
        <button onClick={autoLayout} disabled={demoMode}>自动排列</button>
        <button onClick={resetDocument} disabled={demoMode}>恢复示例</button>
        <button onClick={() => loadTeachingCase('direct-answer')} disabled={demoMode}>案例 1：直接回答</button>
        <button onClick={() => loadTeachingCase('tool-call')} disabled={demoMode}>案例 2：调用工具</button>
        <button onClick={() => loadTeachingCase('file-summary')} disabled={demoMode}>案例 3：搜索并总结文件</button>
        <button onClick={() => loadTeachingCase('file-choice')} disabled={demoMode}>案例 4：搜索后询问文件</button>
        <button onClick={exportDocument}>导出 JSON</button>
      </div>

      {navigationPanels && <div style={{ marginTop: 14 }}>{navigationPanels}</div>}
      </nav>

      {!demoMode && (
        <button
          onClick={() => { void runCurrentWorkflow(); }}
          disabled={disabled || !onRun}
          style={{ position: 'fixed', zIndex: 25, left: '50%', bottom: 18, transform: 'translateX(-50%)', padding: '11px 20px', border: '1px solid #1c5fd4', borderRadius: 9, background: '#1769e0', color: '#ffffff', boxShadow: '0 6px 18px #1769e052', fontWeight: 800, whiteSpace: 'nowrap' }}
        >
          运行当前工作流
        </button>
      )}

      {demoMode && demoRunStatus === 'waiting_user' && (
        <form
          onSubmit={(event) => { event.preventDefault(); void submitWaitingInput(); }}
          style={{ position: 'fixed', zIndex: 35, top: 18, left: '50%', transform: 'translateX(-50%)', width: 'min(560px, calc(100vw - 32px))', boxSizing: 'border-box', padding: 14, border: '1px solid #d9aa4a', borderRadius: 11, background: '#fffaf0', boxShadow: '0 10px 28px #1720332a' }}
        >
          <div style={{ color: '#8a5a00', fontWeight: 800, fontSize: 13 }}>LLM 决策：ask_user · 等待你的补充</div>
          <div style={{ marginTop: 6, color: '#3f4652', fontSize: 14, lineHeight: 1.45 }}>{visibleWaitingQuestion}</div>
          {waitingChoices.length > 0 && (
            <div style={{ display: 'grid', gap: 6, marginTop: 10 }}>
              <div style={{ color: '#697386', fontSize: 12 }}>{multipleFileSelection ? `从已搜索到的文件中勾选 ${waitingMinSelections}-${waitingMaxSelections} 个（LLM 请求多选）：` : '从已搜索到的文件中选择（LLM 请求单选）：'}</div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {waitingChoices.map((choice) => (
                  <button
                    key={choice.file_id}
                    type="button"
                    disabled={waitingSubmitting || !onSubmitUserInput}
                    onClick={() => {
                      if (!multipleFileSelection) { void submitWaitingInput(choice.name, choice.file_id); return; }
                      setWaitingSelectionIds((current) => current.includes(choice.file_id) ? current.filter((fileId) => fileId !== choice.file_id) : current.length >= waitingMaxSelections ? current : [...current, choice.file_id]);
                    }}
                    title={choice.directory ? `${choice.directory}/${choice.name}` : choice.name}
                    style={{ padding: '6px 9px', border: '1px solid #d9aa4a', borderRadius: 7, background: multipleFileSelection && activeWaitingSelectionIds.includes(choice.file_id) ? '#ffe9b5' : '#fff', color: '#6e4a08', fontSize: 12, fontWeight: 700 }}
                  >
                    {multipleFileSelection && activeWaitingSelectionIds.includes(choice.file_id) ? '✓ ' : ''}{choice.name}{choice.directory ? ` · ${choice.directory}` : ''}
                  </button>
                ))}
              </div>
              {multipleFileSelection && <button type="button" onClick={() => void submitWaitingInput(activeWaitingSelectionIds.map((fileId) => waitingChoices.find((choice) => choice.file_id === fileId)?.name || fileId).join('、'), undefined, activeWaitingSelectionIds)} disabled={waitingSubmitting || !onSubmitUserInput || activeWaitingSelectionIds.length < waitingMinSelections} style={{ justifySelf: 'start', padding: '6px 10px', fontSize: 12 }}>确认 {activeWaitingSelectionIds.length} 个文件并续跑</button>}
            </div>
          )}
          {allowsCustomWaitingInput && <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <input
              autoFocus
              value={waitingInput}
              onChange={(event) => setWaitingInput(event.target.value)}
              placeholder={waitingChoices.length ? multipleFileSelection ? '或用逗号分隔输入多个完整/唯一文件名' : '或输入完整/部分文件名' : '输入补充信息后继续运行'}
              disabled={waitingSubmitting || !onSubmitUserInput}
              style={{ ...inputStyle, flex: 1, minWidth: 0 }}
            />
            <button type="submit" disabled={!waitingInput.trim() || waitingSubmitting || !onSubmitUserInput} style={{ padding: '8px 12px', whiteSpace: 'nowrap' }}>{waitingSubmitting ? '续跑中…' : '提交并续跑'}</button>
          </div>}
          {waitingSubmitError && <div style={{ marginTop: 8, color: '#a33a2b', fontSize: 12 }}>{waitingSubmitError}</div>}
        </form>
      )}

      {demoMode && demoEvents.length > 0 && (
        <>
        {playbackPanelCollapsed ? (
          <div style={{ position: 'fixed', zIndex: 30, left: 12, right: 12, bottom: 0, padding: '6px 12px', border: '1px solid #c7d8f2', borderBottom: 0, borderRadius: '9px 9px 0 0', background: '#f4f8ff', boxShadow: '0 -4px 18px #1720331a', textAlign: 'center' }}>
            <button onClick={() => setPlaybackPanelCollapsed(false)}>展开画布回放控制</button>
          </div>
        ) : (
        <div style={{ position: 'fixed', zIndex: 30, left: 12, right: 12, bottom: 12, maxHeight: '42vh', overflowY: 'auto', margin: 0, padding: 12, border: '1px solid #c7d8f2', borderRadius: 9, background: '#f4f8ff', boxShadow: '0 -4px 18px #1720331a' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <strong>画布回放控制</strong>
              <span style={{ padding: '4px 8px', borderRadius: 999, border: `1px solid ${statusPresentation.border}`, color: statusPresentation.color, background: statusPresentation.background, fontSize: 12, fontWeight: 700 }}>{statusPresentation.label}</span>
              {demoRunId && <span style={{ color: '#697386', fontSize: 11 }}>运行 ID：{demoRunId}</span>}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 12, color: '#4d586a' }}>
              {demoCursor === null
                  ? `实时跟随 · ${demoEvents.length} 条事件`
                  : demoCursor === 0
                    ? `事件 0 / ${demoEvents.length} · 运行尚未开始`
                    : `事件 ${demoCursor} / ${demoEvents.length} · 第 ${latestDemoEvent?.step ?? 0} 步 · ${latestDemoEvent?.type ?? ''}`}
              </span>
              <button onClick={() => setPlaybackPanelCollapsed(true)} style={{ padding: '4px 8px', fontSize: 12 }}>收起</button>
            </div>
          </div>
          <input
            aria-label="画布回放事件游标"
            type="range"
            min="0"
            max={demoEvents.length}
            value={demoCursor ?? demoEvents.length}
            onChange={(event) => moveDemoCursor(Number(event.target.value))}
            style={{ display: 'block', width: '100%', margin: '9px 0' }}
          />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(90px, 1fr))', gap: 7, marginBottom: 10 }}>
            {[
              { label: '运行时间', value: terminalDemo ? totalElapsed : liveElapsed, help: '从 run.start 到当前实时事件或终止事件的总耗时，包含模型、工具与 Harness 开销。' },
              { label: '当前步骤', value: `${latestDemoEvent?.step ?? 0} / ${loopModule?.config.max_steps || document.max_steps}`, help: '当前 ReAct 决策步数 / 配置的最大执行步数。一次模型决策会推进一步。' },
              { label: '模型决策', value: demoStats.decisions, help: '收到并解析的 LLM 行动计划总数，包括工具调用、回答、追问与路由。' },
              { label: '回答', value: demoStats.finalAnswerDecisions, help: 'LLM 选择 final_answer 的次数，表示它认为已有足够证据输出结果。' },
              { label: '追问', value: demoStats.askUserDecisions, help: 'LLM 选择 ask_user 的次数，表示它请求用户补充信息后暂停运行。' },
              { label: '工具计划', value: demoStats.toolPlanned, help: 'LLM 在行动计划中提出的工具调用数量；一条 call_tools 可包含多个工具。' },
              { label: '校验通过', value: demoStats.toolValidated, help: '通过 Harness Schema、能力、权限与参数校验的工具调用数量。' },
              { label: '开始执行', value: demoStats.toolStarted, help: '已实际交给工具执行器的调用数量；仅计划或被拦截的调用不计入。' },
              { label: '观察', value: demoStats.observations, help: '工具完成后写回 Context 的 Observation 数量。' },
              { label: '决策错误', value: `${demoStats.decisionErrors} / ${decisionFailureLimit}`, help: 'LLM 输出未能形成合法行动的次数 / 最大决策错误预算。' },
              { label: '工具失败', value: `${demoStats.toolErrors} / ${toolFailureLimit}`, help: '行动通过校验后，工具在真实执行阶段失败的次数 / 最大工具失败预算。' },
              { label: '策略拦截', value: demoStats.policyBlocks, help: '被权限、风险确认或其他安全策略阻止、尚未执行的工具调用数量。' },
            ].map(({ label, value, help }) => (
              <div key={label} title={help} style={{ padding: '7px 8px', border: '1px solid #dce6f4', borderRadius: 7, background: '#ffffff', cursor: 'help' }}>
                <div style={{ fontSize: 11, color: '#697386' }}>{label} <span aria-label={`${label}说明`}>ⓘ</span></div>
                <strong style={{ color: '#172033', fontSize: 16 }}>{value}</strong>
              </div>
            ))}
          </div>
          {Object.keys(demoStats.decisionErrorsByKind).length > 0 && (
            <p style={{ margin: '-2px 0 10px', color: '#a33a2b', fontSize: 12 }}>
              决策错误分类：{Object.entries(demoStats.decisionErrorsByKind).map(([kind, count]) => `${decisionReasonLabels[kind] || kind} ${count}`).join(' · ')}
            </p>
          )}
          {visibleFailureReason && (
            <div style={{ marginBottom: 10, padding: '8px 10px', border: '1px solid #df8578', borderRadius: 7, background: '#fff0ee', color: '#8f2e23', fontSize: 12, lineHeight: 1.45 }}>
              <strong>运行失败原因：</strong>{visibleFailureReason}
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <button onClick={() => moveDemoCursor((demoCursor ?? demoEvents.length) - 1)} disabled={(demoCursor ?? demoEvents.length) === 0}>上一步</button>
            <button onClick={() => {
              if (demoCursor === null || demoCursor >= demoEvents.length) moveDemoCursor(0, false);
              setDemoPlaying((playing) => !playing);
            }}>
              {demoPlaying ? '暂停播放' : '自动播放'}
            </button>
            <button onClick={() => moveDemoCursor((demoCursor ?? demoEvents.length) + 1)} disabled={(demoCursor ?? demoEvents.length) >= demoEvents.length}>下一步</button>
            <button onClick={() => { setDemoCursor(null); setDemoPlaying(false); }} disabled={demoCursor === null}>回到实时末尾</button>
            <button onClick={() => { void runCurrentWorkflow(); }} disabled={disabled || !onRun}>再次运行</button>
            <button onClick={onExitDemo} disabled={!onExitDemo}>退出演示，返回编辑</button>
            <label style={{ color: '#4d586a', fontSize: 12 }}>
              速度
              <select value={demoSpeed} onChange={(event) => setDemoSpeed(Number(event.target.value))} style={{ marginLeft: 5, padding: 4 }}>
                <option value={0.5}>0.5×</option>
                <option value={1}>1×</option>
                <option value={2}>2×</option>
                <option value={4}>4×</option>
              </select>
            </label>
          </div>
        </div>
        )}
        </>
      )}

      <div style={{ position: 'absolute', inset: 0 }}>
        <aside style={{ position: 'fixed', zIndex: 20, top: 0, left: 0, width: 'min(300px, 88vw)', height: '100vh', overflowY: 'auto', boxSizing: 'border-box', border: '1px solid #e0e5ee', borderRadius: '0 12px 12px 0', padding: 16, background: '#fbfcff', boxShadow: '8px 0 28px #17203322', transform: paletteOpen ? 'translateX(0)' : 'translateX(-105%)', transition: 'transform 220ms ease', pointerEvents: paletteOpen ? 'auto' : 'none' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}><strong style={{ fontSize: 14 }}>模块库</strong><button onClick={() => setPaletteOpen(false)} aria-label="关闭模块库">关闭</button></div>
          <p style={{ color: '#697386', fontSize: 12, lineHeight: 1.4 }}>拖动或点击添加模块</p>
          <div style={{ display: 'grid', gap: 8 }}>
            {palette.map((item) => {
              const colors = nodeColors[item.type];
              return (
                <button
                  key={item.type}
                  draggable
                  onDragStart={(event) => handlePaletteDragStart(event, item.type)}
                  onClick={() => addNode(item.type)}
                  style={{
                    textAlign: 'left',
                    padding: 9,
                    border: `1px solid ${colors.border}`,
                    borderRadius: 8,
                    background: colors.background,
                    cursor: 'grab',
                  }}
                >
                  <div style={{ color: colors.label, fontWeight: 700, fontSize: 13 }}>{item.label}</div>
                  <div style={{ color: '#697386', fontSize: 11, marginTop: 4 }}>{item.description}</div>
                </button>
              );
            })}
          </div>
          <div style={{ marginTop: 14, paddingTop: 10, borderTop: '1px solid #e0e5ee' }}>
            <strong style={{ fontSize: 13 }}>工具注册表</strong>
            <p style={{ color: '#697386', fontSize: 12, lineHeight: 1.4, margin: '5px 0 8px' }}>工具是独立能力；由 Context 注入定义，再由 LLM 自主调用。</p>
            <button
              draggable
              onDragStart={(event) => handlePaletteDragStart(event, toolPalette.type)}
              onClick={() => addNode(toolPalette.type)}
              style={{ textAlign: 'left', padding: 9, border: `1px solid ${nodeColors.tool.border}`, borderRadius: 8, background: nodeColors.tool.background, cursor: 'grab', width: '100%' }}
            >
              <div style={{ color: nodeColors.tool.label, fontWeight: 700, fontSize: 13 }}>{toolPalette.label}</div>
              <div style={{ color: '#697386', fontSize: 11, marginTop: 4 }}>{toolPalette.description}</div>
            </button>
          </div>
          <div style={{ marginTop: 14, paddingTop: 10, borderTop: '1px solid #e0e5ee', color: '#697386', fontSize: 12, lineHeight: 1.5 }}>
            Context 可包含用户输入、提示词和工具定义模块；按 Delete 可删除选中模块。
          </div>
        </aside>

        <div
          onContextMenu={(event) => { event.preventDefault(); setCanvasMenu({ x: event.clientX, y: event.clientY }); }}
          style={{
            position: 'absolute',
            inset: 0,
            overflow: 'hidden',
            backgroundColor: '#fbfcff',
            backgroundImage: 'linear-gradient(#e8edf5 1px, transparent 1px), linear-gradient(90deg, #e8edf5 1px, transparent 1px)',
            backgroundSize: `${24 * canvasZoom}px ${24 * canvasZoom}px`,
            backgroundPosition: `${canvasPan.x}px ${canvasPan.y}px`,
          }}
        >
          <div style={{ position: 'absolute', left: canvasPan.x, top: canvasPan.y, width: canvasWidth * canvasZoom, height: canvasHeight * canvasZoom }}>
          <div
            ref={canvasRef}
            onDragOver={(event) => event.preventDefault()}
            onDrop={handleCanvasDrop}
            onWheel={handleCanvasWheel}
            onPointerDown={startCanvasPan}
            onClick={(event) => {
              if (event.target === event.currentTarget) {
                setSelectedNodeId(null);
                setCanvasMenu(null);
              }
            }}
            style={{
              position: 'relative',
              width: canvasWidth,
              height: canvasHeight,
              minWidth: '100%',
              transform: `scale(${canvasZoom})`,
              transformOrigin: 'top left',
              background: 'transparent',
            }}
          >
            <svg width={canvasWidth} height={canvasHeight} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
              <defs>
                <marker id="editor-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                  <path d="M0,0 L8,4 L0,8 z" fill="#8995a8" />
                </marker>
              </defs>
              {document.edges.map((edge) => {
                const from = nodeMap.get(edge.from);
                const to = nodeMap.get(edge.to);
                if (!from || !to || to.parentId) return null;
                // User input is represented by its explicit Context reference,
                // not by a vague arrow to the outer ReAct container.
                if (contextUserInputModule && loopModule && edge.from === contextUserInputModule.id && edge.to === loopModule.id) return null;
                return (
                  <g key={edge.id}>
                    <line
                      x1={from.x + renderedNodeWidth(from)}
                      y1={from.y + renderedNodeHeight(from) / 2}
                      x2={to.x}
                      y2={to.y + renderedNodeHeight(to) / 2}
                      stroke="#8995a8"
                      strokeWidth="2"
                      markerEnd="url(#editor-arrow)"
                    />
                    <circle
                      cx={(from.x + renderedNodeWidth(from) + to.x) / 2}
                      cy={(from.y + renderedNodeHeight(from) / 2 + to.y + renderedNodeHeight(to) / 2) / 2}
                      r={edge.label ? "15" : "8"}
                      fill="#ffffff"
                      stroke="#8995a8"
                      style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                      onClick={() => deleteEdge(edge.id)}
                    />
                    <text
                      x={(from.x + renderedNodeWidth(from) + to.x) / 2}
                      y={(from.y + renderedNodeHeight(from) / 2 + to.y + renderedNodeHeight(to) / 2) / 2 + 4}
                      textAnchor="middle"
                      fontSize="10"
                      fill="#697386"
                      style={{ pointerEvents: 'none' }}
                    >
                      {edge.label ? `${edge.label} ×` : '×'}
                    </text>
                  </g>
                );
              })}
            </svg>

            {demoMode && (
              <div style={{ position: 'absolute', left: 14, top: 30, zIndex: 3, padding: '8px 11px', borderRadius: 8, background: '#172033', color: '#ffffff', fontSize: 12, boxShadow: '0 4px 14px #17203344' }}>
                <strong>{demoCursor === null ? '实时演示' : '历史回放'}</strong> · 第 {latestDemoEvent?.step ?? 0} 步 · {demoPhase}
              </div>
            )}

            {canvasVisibleNodes.map((node) => {
              const colors = nodeColors[node.type];
              const selected = selectedNodeId === node.id;
              const containedModules = document.nodes.filter((item) => item.parentId === node.id);
              const reactContextModule = node.type === 'react_loop' ? containedModules.find((item) => item.type === 'input') : null;
              const reactLLMModule = node.type === 'react_loop' ? containedModules.find((item) => item.type === 'agent') : null;
              const contextChildren = reactContextModule ? document.nodes.filter((item) => item.parentId === reactContextModule.id) : [];
              const runtimeFocused = demoFocusIds.includes(node.id) || (node.type === 'react_loop' && containedModules.some((item) => demoFocusIds.includes(item.id)));
              const connectable = ['user_input', 'input', 'react_loop', 'agent', 'condition', 'ask_user', 'final'].includes(node.type);
              return (
                <div
                  key={node.id}
                  data-workflow-module="true"
                  onPointerDown={(event) => startMoving(event, node)}
                  onDragOver={(event) => {
                    if (node.type === 'input' || node.type === 'react_loop') event.preventDefault();
                  }}
                  onDrop={(event) => {
                    if (node.type === 'input' || node.type === 'react_loop') handleContainerDrop(event, node);
                  }}
                  onClick={() => {
                    if (suppressNodeClickRef.current === node.id) {
                      suppressNodeClickRef.current = null;
                      return;
                    }
                    if (!demoMode) selectNode(node.id);
                  }}
                  style={{
                    position: 'absolute',
                    left: node.x,
                    top: node.y,
                    width: renderedNodeWidth(node),
                    minHeight: renderedNodeHeight(node),
                    boxSizing: 'border-box',
                    padding: 12,
                    border: `2px solid ${colors.border}`,
                    borderRadius: 10,
                    background: colors.background,
                    boxShadow: runtimeFocused ? '0 0 0 5px #ffbf47aa, 0 0 22px #ffbf47' : selected ? `0 0 0 3px ${colors.border}55` : undefined,
                    transform: runtimeFocused ? 'translateY(-3px)' : undefined,
                    transition: 'box-shadow 180ms ease, transform 180ms ease',
                    cursor: movingRef.current?.id === node.id ? 'grabbing' : 'grab',
                    userSelect: 'none',
                    zIndex: 1,
                  }}
                >
                  {connectable && !demoMode && <button
                    data-handle="source"
                    aria-label={`从 ${node.label} 建立连线`}
                    onPointerDown={(event) => startConnection(event, node.id)}
                    onClick={(event) => {
                      event.stopPropagation();
                      setSelectedNodeId(node.id);
                      setConnectingFrom(node.id);
                    }}
                    style={{ position: 'absolute', right: -8, top: '50%', transform: 'translateY(-50%)', width: 16, height: 16, borderRadius: '50%', border: `2px solid ${colors.border}`, background: '#ffffff', cursor: 'crosshair', padding: 0 }}
                  />}
                  {connectable && !demoMode && <button
                    data-handle="target"
                    aria-label={`连接到 ${node.label}`}
                    onPointerUp={(event) => finishConnection(event, node.id)}
                    onClick={(event) => finishConnection(event, node.id)}
                    style={{ position: 'absolute', left: -8, top: '50%', transform: 'translateY(-50%)', width: 16, height: 16, borderRadius: '50%', border: `2px solid ${colors.border}`, background: '#ffffff', cursor: 'crosshair', padding: 0 }}
                  />}
                  <div style={{ color: colors.label, fontWeight: 700, fontSize: 13 }}>{node.label}</div>
                  {node.type === 'react_loop' ? (
                    <ReActRuntimeDiagram
                      playback={reactPlayback}
                      demoMode={demoMode}
                      onSelectContext={!demoMode && reactContextModule ? () => setSelectedNodeId(reactContextModule.id) : undefined}
                      onSelectLLM={!demoMode && reactLLMModule ? () => setSelectedNodeId(reactLLMModule.id) : undefined}
                      embeddedModules={contextChildren.filter((child) => child.type !== 'system_prompt' && child.type !== 'tool_definition')}
                      onSelectModule={!demoMode ? (nodeId) => setSelectedNodeId(nodeId) : undefined}
                      userInputModule={contextUserInputModule}
                      systemPromptModules={contextSystemPromptModules}
                      toolDefinitionModules={contextToolDefinitionModules}
                      contextWindowTokens={contextWindowTokens}
                    />
                  ) : (
                    <div style={{ marginTop: 8, color: '#4d586a', fontSize: 12, lineHeight: 1.35, wordBreak: 'break-word' }}>
                      {node.type === 'final' && demoMode && visibleFinalOutput ? (
                        <div>{visibleFinalOutput}</div>
                      ) : node.type === 'ask_user' && demoMode && waitingEvent ? (
                        <>
                          <div style={{ marginBottom: 4, color: colors.label, fontWeight: 700, fontSize: 11 }}>本轮 LLM 实际追问</div>
                          <div>{visibleWaitingQuestion}</div>
                        </>
                      ) : compactContent(node, toolModules)}
                    </div>
                  )}
                  {node.type === 'input' && (
                    <div style={{ marginTop: 14, paddingTop: 10, borderTop: '1px dashed #9db7de' }}>
                      <div style={{ color: colors.label, fontWeight: 700, fontSize: 11, marginBottom: 7 }}>包含的模块（组装为每轮模型 Context）</div>
                      {containedModules.length === 0 ? <div style={{ color: '#697386', fontSize: 11 }}>可从模块库添加系统提示词或工具定义模块。</div> : (
                        <div style={{ display: 'grid', gap: 6 }}>
                          {containedModules.map((child) => (
                            <div key={child.id} onPointerDown={(event) => event.stopPropagation()} style={{ textAlign: 'left', padding: 7, border: demoFocusIds.includes(child.id) ? '2px solid #ffbf47' : '1px solid #9db7de', borderRadius: 6, background: demoFocusIds.includes(child.id) ? '#fff8df' : '#ffffff', color: '#315e9e', fontSize: 11, transition: 'all 180ms ease' }}>
                              <button onClick={(event) => { event.stopPropagation(); setSelectedNodeId(child.id); }} style={{ border: 0, padding: 0, background: 'transparent', color: 'inherit', textAlign: 'left', cursor: 'pointer', font: 'inherit', width: '100%' }}>
                                {child.label} · {compactContent(child, toolModules)}
                              </button>
                              {child.type === 'input' && document.nodes.filter((item) => item.parentId === child.id).map((grandchild) => (
                                <button key={grandchild.id} onClick={(event) => { event.stopPropagation(); setSelectedNodeId(grandchild.id); }} style={{ display: 'block', marginTop: 5, width: '100%', border: '1px dashed #b6cce5', borderRadius: 5, padding: 5, background: '#f8fbff', color: '#315e9e', textAlign: 'left', fontSize: 10 }}>
                                  ↳ {grandchild.label} · {compactContent(grandchild, toolModules)}
                                </button>
                              ))}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
            <svg width={canvasWidth} height={canvasHeight} style={{ position: 'absolute', inset: 0, zIndex: 3, pointerEvents: 'none', overflow: 'visible' }}>
              <defs>
                <marker id="editor-overlay-arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 z" fill="#5b6475" /></marker>
              </defs>
              {loopModule && contextUserInputModule && contextReferenceAnchors[contextUserInputModule.id] && (
                <g>
                  <path
                    d={`M ${contextUserInputModule.x + renderedNodeWidth(contextUserInputModule)} ${contextUserInputModule.y + renderedNodeHeight(contextUserInputModule) / 2} C ${contextUserInputModule.x + renderedNodeWidth(contextUserInputModule) + 44} ${contextUserInputModule.y + renderedNodeHeight(contextUserInputModule) / 2}, ${contextReferenceAnchors[contextUserInputModule.id].x - 44} ${contextReferenceAnchors[contextUserInputModule.id].y}, ${contextReferenceAnchors[contextUserInputModule.id].x} ${contextReferenceAnchors[contextUserInputModule.id].y}`}
                    fill="none" stroke="#297fca" strokeWidth="2.5" strokeDasharray="5 4" markerEnd="url(#editor-overlay-arrow)"
                  />
                </g>
              )}
              {loopModule && contextSystemPromptModules.map((prompt) => contextReferenceAnchors[prompt.id] && (
                <g key={`context-system-link-front-${prompt.id}`}>
                  <path
                    d={`M ${prompt.x + renderedNodeWidth(prompt)} ${prompt.y + renderedNodeHeight(prompt) / 2} C ${prompt.x + renderedNodeWidth(prompt) + 44} ${prompt.y + renderedNodeHeight(prompt) / 2}, ${contextReferenceAnchors[prompt.id].x - 44} ${contextReferenceAnchors[prompt.id].y}, ${contextReferenceAnchors[prompt.id].x} ${contextReferenceAnchors[prompt.id].y}`}
                    fill="none" stroke="#5478bc" strokeWidth="2.5" strokeDasharray="5 4" markerEnd="url(#editor-overlay-arrow)"
                  />
                </g>
              ))}
              {loopModule && contextToolDefinitionModules.map((definition) => contextReferenceAnchors[definition.id] && (
                <g key={`context-tool-definition-link-front-${definition.id}`}>
                  <path
                    d={`M ${definition.x} ${definition.y + renderedNodeHeight(definition) / 2} C ${definition.x - 44} ${definition.y + renderedNodeHeight(definition) / 2}, ${contextReferenceAnchors[definition.id].x + 44} ${contextReferenceAnchors[definition.id].y}, ${contextReferenceAnchors[definition.id].x} ${contextReferenceAnchors[definition.id].y}`}
                    fill="none" stroke="#d69416" strokeWidth="2.5" strokeDasharray="5 4" markerEnd="url(#editor-overlay-arrow)"
                  />
                </g>
              ))}
              {contextToolDefinitionModules.flatMap((definition) => selectedModuleIds(definition).map((toolId) => ({ definition, tool: toolModules.find((tool) => tool.id === toolId) }))).map(({ definition, tool }) => tool && (
                <g key={`tool-definition-tool-link-${definition.id}-${tool.id}`}>
                  <path
                    d={`M ${definition.x + renderedNodeWidth(definition)} ${definition.y + renderedNodeHeight(definition) / 2} C ${definition.x + renderedNodeWidth(definition) + 34} ${definition.y + renderedNodeHeight(definition) / 2}, ${tool.x - 34} ${tool.y + renderedNodeHeight(tool) / 2}, ${tool.x} ${tool.y + renderedNodeHeight(tool) / 2}`}
                    fill="none" stroke="#d69416" strokeWidth="2" strokeDasharray="5 4" markerEnd="url(#editor-overlay-arrow)"
                  />
                </g>
              ))}
            </svg>
            {connectingFrom && (
              <div style={{ position: 'absolute', left: 12, bottom: 12, padding: '5px 8px', borderRadius: 6, background: '#172033', color: '#ffffff', fontSize: 12, zIndex: 2 }}>
                正在连线：请点击目标节点左侧圆点
              </div>
            )}
          </div>
          </div>
        </div>

        {canvasMenu && (
          <div style={{ position: 'fixed', zIndex: 35, left: canvasMenu.x, top: canvasMenu.y, minWidth: 172, padding: 6, border: '1px solid #cfd8e6', borderRadius: 8, background: '#ffffff', boxShadow: '0 8px 24px #1720332b' }}>
            <button onClick={centerCanvas} style={{ width: '100%', border: 0, borderRadius: 5, padding: '8px 10px', background: 'transparent', color: '#253a55', textAlign: 'left', cursor: 'pointer' }}>回到画布中心</button>
          </div>
        )}

        <aside style={{ position: 'fixed', zIndex: 20, top: 0, right: 0, width: 'min(360px, 92vw)', height: '100vh', overflowY: 'auto', boxSizing: 'border-box', border: '1px solid #e0e5ee', borderRadius: '12px 0 0 12px', padding: 16, background: '#ffffff', boxShadow: '-8px 0 28px #17203322', transform: selectedNode ? 'translateX(0)' : 'translateX(105%)', transition: 'transform 220ms ease', pointerEvents: selectedNode ? 'auto' : 'none' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}><strong>节点属性</strong><button onClick={() => setSelectedNodeId(null)} aria-label="关闭节点属性">关闭</button></div>
          {!selectedNode ? (
            <p style={{ color: '#697386', fontSize: 13 }}>点击画布中的节点编辑属性。</p>
          ) : (
            <div style={{ display: 'grid', gap: 10, marginTop: 10 }}>
              <label style={{ fontSize: 12, color: '#4d586a' }}>
                节点名称
                <input value={selectedNode.label} onChange={(event) => updateSelected({ label: event.target.value })} style={{ ...inputStyle, marginTop: 4 }} />
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <label style={{ fontSize: 12, color: '#4d586a' }}>
                  模块宽度
                  <input
                    type="number"
                    min={selectedNode.type === 'input' ? 320 : selectedNode.type === 'react_loop' ? 620 : 130}
                    max="1600"
                    value={renderedNodeWidth(selectedNode)}
                    onChange={(event) => updateSelected({ width: Number(event.target.value) || renderedNodeWidth(selectedNode) })}
                    style={{ ...inputStyle, marginTop: 4 }}
                  />
                </label>
                <label style={{ fontSize: 12, color: '#4d586a' }}>
                  模块高度
                  <input
                    type="number"
                    min={selectedNode.type === 'input' ? 240 : selectedNode.type === 'react_loop' ? 450 : 88}
                    max="1200"
                    value={renderedNodeHeight(selectedNode)}
                    onChange={(event) => updateSelected({ height: Number(event.target.value) || renderedNodeHeight(selectedNode) })}
                    style={{ ...inputStyle, marginTop: 4 }}
                  />
                </label>
              </div>
              <div style={{ marginTop: -4, color: '#697386', fontSize: 11 }}>尺寸会与模块位置一同自动保存到本地浏览器。</div>
              {selectedNode.type === 'workspace' && (
                <>
                  <div style={{ display: 'grid', gap: 7 }}>
                    <button onClick={() => void chooseWorkspaceDirectory()} disabled={workspaceLoading} style={{ border: '1px solid #78a764', borderRadius: 7, padding: '8px 10px', background: workspaceLoading ? '#edf3ea' : '#f1f8ee', color: '#356c2a', cursor: workspaceLoading ? 'wait' : 'pointer', fontWeight: 700 }}>
                      {workspaceLoading ? '正在打开或读取文件夹…' : selectedNode.config.root_path ? '重新选择文件夹' : '选择教学工作区文件夹'}
                    </button>
                    {selectedNode.config.root_path ? (
                      <div style={{ padding: '7px 8px', border: '1px solid #c9dcc1', borderRadius: 6, background: '#f7fbf5', color: '#356c2a', fontSize: 12, wordBreak: 'break-all' }}>
                        已授权根目录：{selectedNode.config.root_path}
                      </div>
                    ) : <div style={{ color: '#697386', fontSize: 12 }}>尚未选择本机目录，当前将使用内置教学文件。</div>}
                    {selectedNode.config.root_path && <button onClick={() => void refreshWorkspaceTree()} disabled={workspaceLoading} style={{ justifySelf: 'start', border: '1px solid #c9dcc1', borderRadius: 6, padding: '5px 8px', background: '#ffffff', color: '#356c2a', cursor: workspaceLoading ? 'wait' : 'pointer', fontSize: 12 }}>刷新文件树</button>}
                  </div>
                  <p style={{ margin: 0, color: '#697386', fontSize: 12, lineHeight: 1.5 }}>
                    选择后，搜索文件只能访问此根目录下、由用户任务指定的子目录；读取文件仍必须使用搜索结果返回的 file_id。当前支持 .txt、.md、.csv、.json、.log 等文本文件。
                  </p>
                  {workspaceError && <div style={{ padding: '7px 8px', border: '1px solid #df8578', borderRadius: 6, background: '#fff0ee', color: '#a33a2b', fontSize: 12 }}>文件树读取失败：{workspaceError}</div>}
                  {workspaceTree && (
                    <div style={{ display: 'grid', gap: 7, paddingTop: 4 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, color: '#356c2a', fontSize: 12, fontWeight: 700 }}>
                        <span>当前目录文件树</span><span style={{ color: '#697386', fontWeight: 400 }}>{workspaceTree.entry_count} 项</span>
                      </div>
                      {workspaceTree.entries.length ? <div style={{ maxHeight: 300, overflowY: 'auto', padding: 8, border: '1px solid #d7e6d1', borderRadius: 7, background: '#fbfdf9' }}><WorkspaceTreeView entries={workspaceTree.entries} /></div> : <div style={{ color: '#697386', fontSize: 12 }}>该目录暂时没有可显示的文件或子目录。</div>}
                      {workspaceTree.truncated && <div style={{ color: '#8b6b45', fontSize: 11 }}>为保持画布响应，文件树最多显示 {workspaceTree.max_entries} 项、向下 {workspaceTree.max_depth} 层。</div>}
                    </div>
                  )}
                </>
              )}
              {selectedNode.type === 'user_input' && (
                <label style={{ fontSize: 12, color: '#4d586a' }}>
                  运行开始时的真实用户输入
                  <textarea value={selectedNode.config.text || ''} onChange={(event) => updateConfig('text', event.target.value)} style={{ ...inputStyle, marginTop: 4, minHeight: 80, resize: 'vertical' }} />
                  <span style={{ display: 'block', marginTop: 5, color: '#697386' }}>运行时它会沿连线注入 ReAct Loop 内的 Context；之后各轮 LLM 输出和 Observation 会继续累积在那里。</span>
                </label>
              )}
              {selectedNode.type === 'input' && (
                <>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    所属 ReAct 循环
                    <select value={selectedNode.parentId || ''} onChange={(event) => updateSelected({ parentId: event.target.value || null })} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="">独立 Context（兼容旧图）</option>
                      {document.nodes.filter((node) => node.type === 'react_loop').map((loop) => <option key={loop.id} value={loop.id}>{loop.label}</option>)}
                    </select>
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    本轮用户消息来源
                    <select value={selectedNode.config.user_input_module_id || ''} onChange={(event) => updateConfig('user_input_module_id', event.target.value)} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="">自动选择第一个用户输入模块</option>
                      {document.nodes.filter((node) => node.type === 'user_input').map((input) => <option key={input.id} value={input.id}>{input.label} · {input.config.text || '未填写'}</option>)}
                    </select>
                    <span style={{ display: 'block', marginTop: 5, color: '#697386' }}>该选择决定哪一个用户输入会在运行开始时注入此 Context；它可以是起始模块，也可以来自另一段流程。</span>
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    Context 窗口大小（tokens）
                    <input type="number" min="256" max="131072" step="256" value={selectedNode.config.context_window_size || '8192'} onChange={(event) => updateConfig('context_window_size', String(Math.min(131072, Math.max(256, Number(event.target.value) || 8192))))} style={{ ...inputStyle, marginTop: 4 }} />
                    <span style={{ display: 'block', marginTop: 5, color: '#697386' }}>默认 8192。运行当前工作流时，此值会作为 llama.cpp 的实际 Context 窗口启动参数；画布按相同预算显示占用比例，精确 token 数仍取决于模型 tokenizer。</span>
                  </label>
                  <div style={{ fontSize: 12, color: '#4d586a' }}>
                    已内嵌的上下文模块
                    <div style={{ display: 'grid', gap: 6, marginTop: 7 }}>
                      {document.nodes.filter((node) => node.parentId === selectedNode.id && node.type !== 'system_prompt').map((child) => (
                        <button key={child.id} onClick={() => setSelectedNodeId(child.id)} style={{ padding: '7px 8px', textAlign: 'left', border: '1px solid #b6cce5', borderRadius: 6, background: '#f8fbff', color: '#315e9e', cursor: 'pointer', fontSize: 12 }}>
                          配置 {workflowNodeTypeLabel(child.type)} · {child.label}
                        </button>
                      ))}
                      {document.nodes.every((node) => node.parentId !== selectedNode.id || node.type === 'system_prompt') && <span style={{ color: '#9b6a22' }}>尚未内嵌工具定义。</span>}
                    </div>
                  </div>
                </>
              )}
              {(selectedNode.type === 'system_prompt' || selectedNode.type === 'tool_definition') && (
                <label style={{ fontSize: 12, color: '#4d586a' }}>
                  所属 Context
                  <select value={selectedNode.parentId || ''} onChange={(event) => updateSelected({ parentId: event.target.value || null })} style={{ ...inputStyle, marginTop: 4 }}>
                    <option value="">请选择 Context</option>
                    {contextModules.map((context) => <option key={context.id} value={context.id}>{context.label}</option>)}
                  </select>
                </label>
              )}
              {selectedNode.type === 'system_prompt' && (
                <label style={{ fontSize: 12, color: '#4d586a' }}>
                  提示词内容
                  <textarea value={selectedNode.config.content || ''} onChange={(event) => updateConfig('content', event.target.value)} style={{ ...inputStyle, marginTop: 4, minHeight: 100, resize: 'vertical' }} />
                </label>
              )}
              {selectedNode.type === 'tool_definition' && (
                  <div style={{ fontSize: 12, color: '#4d586a' }}>
                    注入 Context 的工具定义
                  <div style={{ display: 'grid', gap: 6, marginTop: 7 }}>
                    {toolModules.length === 0 ? <span style={{ color: '#9b6a22' }}>请先从工具注册表添加工具模块。</span> : toolModules.map((tool) => (
                      <label key={tool.id} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <input type="checkbox" checked={selectedModuleIds(selectedNode).includes(tool.id)} onChange={(event) => toggleModuleId('tool_ids', tool.id, event.target.checked)} />
                        {tool.label} · {tool.config.tool}
                      </label>
                    ))}
                  </div>
                  {(() => {
                    const ids = selectedModuleIds(selectedNode);
                    const hasRead = toolModules.some((tool) => tool.config.tool === 'read_file' && ids.includes(tool.id));
                    const hasSearch = toolModules.some((tool) => tool.config.tool === 'search_files' && ids.includes(tool.id));
                    return hasRead && !hasSearch ? <span style={{ display: 'block', marginTop: 7, color: '#a33a2b' }}>读取文件依赖“搜索文件”；请同时注册搜索文件，才能取得可读取的 file_id。</span> : null;
                  })()}
                </div>
              )}
              {selectedNode.type === 'agent' && (
                <>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    所属 ReAct 循环
                    <select value={selectedNode.parentId || ''} onChange={(event) => updateSelected({ parentId: event.target.value || null })} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="">独立 LLM（兼容旧图）</option>
                      {document.nodes.filter((node) => node.type === 'react_loop').map((loop) => <option key={loop.id} value={loop.id}>{loop.label}</option>)}
                    </select>
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    推理引擎
                    <select value={selectedNode.config.brain_type || 'test'} onChange={(event) => updateConfig('brain_type', event.target.value)} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="test">test（演示）</option>
                      <option value="tiny_llm">tiny_llm（本地模型）</option>
                      <option value="clarification">clarification（追问）</option>
                    </select>
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    本地模型
                    <select value={selectedNode.config.model || ''} onChange={(event) => void activateModel(event.target.value)} disabled={modelSwitching} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="">使用当前活动模型</option>
                      {selectedNode.config.model && !localModels.some((model) => model.id === selectedNode.config.model) && <option value={selectedNode.config.model}>{selectedNode.config.model} · 当前外部服务</option>}
                      {localModels.map((model) => <option key={model.id} value={model.id}>{model.label}{model.is_instruction_tuned ? '' : ' · Base，非指令微调'}{model.active ? ' · 当前活动' : ''}</option>)}
                    </select>
                    <span style={{ display: 'block', marginTop: 5, color: '#697386' }}>{localModels.length ? '选择后将在独立 llama.cpp 服务中加载该 GGUF；不会修改模型文件。' : '未发现完整 GGUF，请检查模型目录或等待下载完成。'}</span>
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    执行设备
                    <select
                      value={selectedNode.config.device || 'cpu'}
                      disabled={modelSwitching}
                      onChange={(event) => {
                        const device = event.target.value;
                        updateConfig('device', device);
                        if (selectedNode.config.model) void activateModel(selectedNode.config.model, device);
                      }}
                      style={{ ...inputStyle, marginTop: 4 }}
                    >
                      {localDevices.length === 0 && <option value="cpu">CPU</option>}
                      {localDevices.map((device) => <option key={device.id} value={device.id} disabled={!device.available}>{device.label}{device.available ? '' : ' · 未安装运行时'}</option>)}
                    </select>
                    <span style={{ display: 'block', marginTop: 5, color: '#697386' }}>
                      GPU 会用 CUDA 版 llama.cpp 以尽可能多地卸载模型层到显存；当前可用性由本机运行时检测。
                    </span>
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    工具调用协议
                    <select value={selectedNode.config.tool_protocol || 'json_action'} onChange={(event) => updateConfig('tool_protocol', event.target.value)} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="json_action">JSON Action（本地模型兼容 / 教学 DSL）</option>
                      <option value="native_tools">Native Tool Calling（OpenAI tools / tool_calls）</option>
                    </select>
                    <span style={{ display: 'block', marginTop: 4, color: '#697386' }}>原生模式要求推理服务支持 Chat Completions 的 tools 与 tool_calls；兼容模式会明确标注为教学适配层。</span>
                  </label>
                  <fieldset style={{ margin: 0, padding: 10, border: '1px solid #d9e1ed', borderRadius: 7 }}>
                    <legend style={{ padding: '0 4px', color: '#4d586a', fontSize: 12 }}>LLM 采样与请求控制</legend>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
                      <label style={{ fontSize: 12, color: '#4d586a' }}>
                        温度
                        <input aria-label="LLM 温度" type="number" min="0" max="2" step="0.05" value={selectedNode.config.temperature || '0.2'} onChange={(event) => updateConfig('temperature', String(Math.min(2, Math.max(0, Number(event.target.value) || 0))))} style={{ ...inputStyle, marginTop: 4 }} />
                      </label>
                      <label style={{ fontSize: 12, color: '#4d586a' }}>
                        Top‑P
                        <input aria-label="LLM Top P" type="number" min="0" max="1" step="0.05" value={selectedNode.config.top_p || '0.9'} onChange={(event) => updateConfig('top_p', String(Math.min(1, Math.max(0, Number(event.target.value) || 0))))} style={{ ...inputStyle, marginTop: 4 }} />
                      </label>
                      <label style={{ fontSize: 12, color: '#4d586a' }}>
                        Top‑K
                        <input aria-label="LLM Top K" type="number" min="0" max="200" step="1" value={selectedNode.config.top_k || '40'} onChange={(event) => updateConfig('top_k', String(Math.min(200, Math.max(0, Math.round(Number(event.target.value) || 0)))))} style={{ ...inputStyle, marginTop: 4 }} />
                      </label>
                      <label style={{ fontSize: 12, color: '#4d586a' }}>
                        最大输出 tokens
                        <input aria-label="LLM 最大输出 tokens" type="number" min="16" max="4096" step="16" value={selectedNode.config.max_tokens || '512'} onChange={(event) => updateConfig('max_tokens', String(Math.min(4096, Math.max(16, Math.round(Number(event.target.value) || 16)))))} style={{ ...inputStyle, marginTop: 4 }} />
                      </label>
                      <label style={{ fontSize: 12, color: '#4d586a' }}>
                        重复惩罚
                        <input aria-label="LLM 重复惩罚" type="number" min="0" max="2" step="0.05" value={selectedNode.config.repeat_penalty || '1.1'} onChange={(event) => updateConfig('repeat_penalty', String(Math.min(2, Math.max(0, Number(event.target.value) || 0))))} style={{ ...inputStyle, marginTop: 4 }} />
                      </label>
                      <label style={{ fontSize: 12, color: '#4d586a' }}>
                        单次决策请求重试
                        <input aria-label="单次决策请求重试次数" type="number" min="1" max="5" step="1" value={selectedNode.config.max_attempts || '2'} onChange={(event) => updateConfig('max_attempts', String(Math.min(5, Math.max(1, Math.round(Number(event.target.value) || 1)))))} style={{ ...inputStyle, marginTop: 4 }} />
                      </label>
                    </div>
                    <span style={{ display: 'block', marginTop: 7, color: '#697386', fontSize: 11, lineHeight: 1.45 }}>这些值会随每次 tiny_llm 请求传给本地 llama.cpp。温度越低输出越稳定；Top‑P / Top‑K 收窄候选范围；最大输出限制单次生成长度。test（演示）引擎不会使用它们。</span>
                  </fieldset>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    指令
                    <textarea value={selectedNode.config.prompt || ''} onChange={(event) => updateConfig('prompt', event.target.value)} style={{ ...inputStyle, marginTop: 4, minHeight: 80, resize: 'vertical' }} />
                  </label>
                  <div style={{ fontSize: 12, color: '#4d586a' }}>
                    此 LLM 可自主调用的工具
                    <div style={{ display: 'grid', gap: 6, marginTop: 7 }}>
                      {toolModules.length === 0 ? <span style={{ color: '#9b6a22' }}>请先从工具注册表添加工具模块。</span> : toolModules.map((tool) => (
                        <label key={tool.id} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <input type="checkbox" disabled={hasExplicitToolDefinitions && !contextToolIds.has(tool.id)} checked={selectedModuleIds(selectedNode).includes(tool.id)} onChange={(event) => toggleModuleId('tool_ids', tool.id, event.target.checked)} />
                          {tool.label} · {tool.config.tool}{hasExplicitToolDefinitions && !contextToolIds.has(tool.id) ? '（未注入 Context）' : ''}
                        </label>
                      ))}
                    </div>
                    <span style={{ display: 'block', marginTop: 5, color: '#697386' }}>有效能力 = Context 的工具定义 ∩ 此处勾选项。勾选“读取文件”会自动勾选它依赖的“搜索文件”；模型自行决定是否调用及传入什么参数。用 test Brain 时，可输入“并行演示”来触发 calculator + mock_search 的批量调用。</span>
                  </div>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    演示默认路由
                    <select value={selectedNode.config.default_route || ''} onChange={(event) => updateConfig('default_route', event.target.value)} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="">由模型选择下一节点</option>
                      {selectedOutgoing.map((edge) => {
                        const target = nodeMap.get(edge.to);
                        return target?.type === 'agent' ? <option key={target.id} value={target.id}>{target.label}</option> : null;
                      })}
                    </select>
                    <span style={{ display: 'block', marginTop: 4, color: '#697386' }}>仅用于确定性演示；默认情况下模型会在已连接的后继节点中路由。</span>
                  </label>
                </>
              )}
              {selectedNode.type === 'tool' && (
                <>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    工具
                    <select value={selectedNode.config.tool || ''} onChange={(event) => {
                      const tool = event.target.value;
                      const isWriter = tool === 'memory_store';
                      const requiresPriorSearch = tool === 'read_file';
                      setDocument((current) => ({ ...current, nodes: current.nodes.map((node) => node.id === selectedNode.id ? { ...node, config: { ...node.config, tool, risk: isWriter ? 'write' : 'read', requires_confirmation: String(isWriter), parallel_safe: String(!isWriter && !requiresPriorSearch) } } : node) }));
                    }} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="calculator">calculator</option>
                      <option value="mock_search">mock_search</option>
                      <option value="search_files">search_files</option>
                      <option value="read_file">read_file</option>
                      <option value="read_document">read_document</option>
                      <option value="extract_numbers">extract_numbers</option>
                      <option value="memory_store">memory_store</option>
                      <option value="memory_lookup">memory_lookup</option>
                    </select>
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    执行权限
                    <select value={selectedNode.config.permission || 'granted'} onChange={(event) => updateConfig('permission', event.target.value)} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="granted">已授权 · 模型可请求执行</option>
                      <option value="blocked">已禁止 · 拒绝所有调用</option>
                    </select>
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    风险等级
                    <select value={selectedNode.config.risk || 'read'} onChange={(event) => updateConfig('risk', event.target.value)} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="read">read · 只读查询</option>
                      <option value="write">write · 会写入状态</option>
                      <option value="sensitive">sensitive · 敏感操作</option>
                    </select>
                  </label>
                  <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, color: '#4d586a' }}>
                    <input type="checkbox" checked={selectedNode.config.parallel_safe !== 'false'} onChange={(event) => updateConfig('parallel_safe', String(event.target.checked))} />
                    可与其他独立工具并行执行
                  </label>
                  <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, color: '#4d586a' }}>
                    <input type="checkbox" checked={selectedNode.config.requires_confirmation === 'true'} onChange={(event) => updateConfig('requires_confirmation', String(event.target.checked))} />
                    执行前必须征求用户确认
                  </label>
                  <p style={{ margin: 0, color: '#697386', fontSize: 12, lineHeight: 1.45 }}>独立注册，不接入实线流程。模型可在一轮中提出多个调用；解释器先检查权限，再按安全策略决定并行、串行或暂停确认。敏感操作始终需要确认，每条结果都会写回 Context。</p>
                </>
              )}
              {selectedNode.type === 'react_loop' && (
                <>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    最大 ReAct 循环次数
                    <input type="number" min="1" max="60" value={selectedNode.config.max_steps || document.max_steps} onChange={(event) => {
                      const maxSteps = Math.min(60, Math.max(1, Number(event.target.value) || 1));
                      updateConfig('max_steps', String(maxSteps));
                      setDocument((current) => ({ ...current, max_steps: maxSteps }));
                    }} style={{ ...inputStyle, marginTop: 4 }} />
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    最大决策错误次数
                    <input type="number" min="0" max="60" value={selectedNode.config.max_decision_failures || '2'} onChange={(event) => updateConfig('max_decision_failures', String(Math.min(60, Math.max(0, Number(event.target.value) || 0))))} style={{ ...inputStyle, marginTop: 4 }} />
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    最大工具执行失败次数
                    <input type="number" min="0" max="10" value={selectedNode.config.max_tool_failures || '2'} onChange={(event) => updateConfig('max_tool_failures', String(Math.min(10, Math.max(0, Number(event.target.value) || 0))))} style={{ ...inputStyle, marginTop: 4 }} />
                  </label>
                  <p style={{ margin: 0, color: '#697386', fontSize: 12, lineHeight: 1.45 }}>每轮：模型读取 Context，选择 Action；工具的 Observation 再写回 Context。模型协议/校验错误与工具执行错误有独立预算；用户拒绝敏感操作只记“策略拦截”，不消耗两种错误预算。</p>
                </>
              )}
              {selectedNode.type === 'condition' && (
                <>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    判断来源
                    <select value={selectedNode.config.source || 'last_tool_result'} onChange={(event) => updateConfig('source', event.target.value)} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="last_tool_result">最近一次工具结果</option>
                      <option value="input">初始用户输入</option>
                      <option value="last_user_input">最近一次用户输入</option>
                    </select>
                  </label>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    判断规则
                    <select value={selectedNode.config.operator || 'equals'} onChange={(event) => updateConfig('operator', event.target.value)} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="equals">等于</option>
                      <option value="contains">包含</option>
                      <option value="greater_than">大于</option>
                      <option value="less_than">小于</option>
                      <option value="is_truthy">有值 / 为真</option>
                      <option value="is_empty">为空</option>
                    </select>
                  </label>
                  {!['is_truthy', 'is_empty'].includes(selectedNode.config.operator || '') && (
                    <label style={{ fontSize: 12, color: '#4d586a' }}>
                      比较值
                      <input value={selectedNode.config.value || ''} onChange={(event) => updateConfig('value', event.target.value)} style={{ ...inputStyle, marginTop: 4 }} />
                    </label>
                  )}
                  <div style={{ fontSize: 12, color: '#4d586a' }}>
                    分支连线（必须各有一条 true / false）
                    {selectedOutgoing.length === 0 ? <div style={{ color: '#9b6a22', marginTop: 5 }}>请从该节点连出两条线。</div> : selectedOutgoing.map((edge) => (
                      <label key={edge.id} style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6 }}>
                        <select value={edge.label || ''} onChange={(event) => updateEdgeLabel(edge.id, event.target.value)} style={{ ...inputStyle, width: 82 }}>
                          <option value="true">true</option>
                          <option value="false">false</option>
                        </select>
                        <span>→ {nodeMap.get(edge.to)?.label || edge.to}</span>
                      </label>
                    ))}
                  </div>
                </>
              )}
              {selectedNode.type === 'ask_user' && (() => {
                const questionMode = selectedNode.config.question_mode || (selectedNode.config.question ? 'fixed' : 'runtime');
                return <>
                  <label style={{ fontSize: 12, color: '#4d586a' }}>
                    追问来源
                    <select value={questionMode} onChange={(event) => updateConfig('question_mode', event.target.value)} style={{ ...inputStyle, marginTop: 4 }}>
                      <option value="runtime">由 LLM 运行时生成（推荐）</option>
                      <option value="fixed">固定问题（图路由演示）</option>
                    </select>
                  </label>
                  {questionMode === 'runtime' ? (
                    <div style={{ padding: '8px 9px', border: '1px solid #ecd49a', borderRadius: 7, background: '#fffaf0', color: '#6e561d', fontSize: 12, lineHeight: 1.5 }}>
                      LLM 根据当前 Context、工具 Observation 与用户任务决定是否追问及具体问题。运行时会在此模块显示本轮实际追问；Harness 只负责暂停、收集输入和续跑。
                    </div>
                  ) : (
                    <label style={{ fontSize: 12, color: '#4d586a' }}>
                      固定提问内容
                      <textarea value={selectedNode.config.question || ''} onChange={(event) => updateConfig('question', event.target.value)} style={{ ...inputStyle, marginTop: 4, minHeight: 80, resize: 'vertical' }} />
                      <span style={{ display: 'block', marginTop: 5, color: '#697386' }}>仅用于固定图路由的确定性演示；自主 LLM Agent 应使用“由 LLM 运行时生成”。</span>
                    </label>
                  )}
                </>;
              })()}
              {selectedNode.type === 'final' && (
                <label style={{ fontSize: 12, color: '#4d586a' }}>
                  输出说明
                  <textarea value={selectedNode.config.answer || ''} onChange={(event) => updateConfig('answer', event.target.value)} style={{ ...inputStyle, marginTop: 4, minHeight: 80, resize: 'vertical' }} />
                </label>
              )}
              <button onClick={deleteSelected} style={{ color: '#a33a2b' }}>删除当前节点</button>
              <button
                onClick={() => {
                  setSelectedNodeId(selectedNode.id);
                  setConnectingFrom(selectedNode.id);
                  setStatus('请点击目标节点，或点击目标节点左侧圆点');
                }}
                disabled={connectingFrom === selectedNode.id || !['input', 'react_loop', 'agent', 'condition', 'ask_user'].includes(selectedNode.type)}
              >
                从此节点开始连线
              </button>
            </div>
          )}
        </aside>
      </div>

      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginTop: 10, color: '#697386', fontSize: 12 }}>
        <span>{document.nodes.length} 个模块</span>
        <span>{document.edges.length} 条连线</span>
        <span>{status}</span>
        <span>点击连线中间的 × 可删除连线</span>
      </div>
    </section>
  );
}
