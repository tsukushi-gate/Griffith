// CONFIGタブ本体。
// プロジェクトの選択/新規作成、Global Config(TrainConfig全体)の
// スキーマ駆動編集、Optimizer/Monitor/Executor定義の管理、
// プロジェクト保存/開く/別名保存(ブラウザDL/アップロード+サーバCRUD)を行う。
// AppShellへの組み込みは別エージェントが行うため、本ファイルはnamed export
// `ConfigTab` を提供するだけで、自身をどこにも配置しない。

import { useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'sonner';
import { Copy, Download, FolderOpen, Plus, Save, Trash2, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  createProject,
  deleteProject,
  getConfigSchema,
  getProject,
  listProjects,
  updateProject,
  type ProjectResponse,
  type ProjectSummary,
} from '../api/rest';
import { buildSchemaDefaults, type JsonSchema } from './jsonSchema';
import { SchemaForm } from './SchemaForm';
import { PROJECT_INSTANTIATED_EVENT } from '../blocks/TemplateGallery';
import { useNetworkStore } from '../store/networkStore';
import type { IRGraphJson } from '../ir/schema';

// --- Optimizer/Monitor/Executor定義の決め打ちスキーマ ---
// TrainConfig本体のJSON Schemaとは独立に、CONFIGタブが扱う束ね形を
// フロント側で決め打ちする(ProjectSchema.optimizers/monitors/executorsは
// dict[str, Any]相当の自由形式のため)。

const OPTIMIZER_DEF_SCHEMA: JsonSchema = {
  type: 'object',
  properties: {
    network: { type: 'string', title: 'ネットワーク', default: '' },
    dataset: { type: 'string', title: 'データセット', default: '' },
    method: { type: 'string', enum: ['bp', 'ap', 'bb'], title: '学習手法', default: 'bp' },
    loss: { type: 'string', title: '損失関数', default: '' },
    optimizer: { type: 'string', enum: ['adam', 'sgd'], title: 'Optimizer種別', default: 'adam' },
    lr: { type: 'number', title: '学習率(lr)', default: 0.01, exclusiveMinimum: 0 },
    logic_lr: { anyOf: [{ type: 'number' }, { type: 'null' }], title: 'Logic LR', default: null },
    ann_lr: { anyOf: [{ type: 'number' }, { type: 'null' }], title: 'ANN LR', default: null },
    batch_size: { type: 'integer', title: 'バッチサイズ', default: 128, minimum: 1 },
    epochs: { type: 'integer', title: 'エポック数', default: 3, minimum: 1 },
  },
};

const MONITOR_DEF_SCHEMA: JsonSchema = {
  type: 'object',
  properties: {
    network: { type: 'string', title: 'ネットワーク', default: '' },
    dataset: { type: 'string', title: 'データセット', default: '' },
    metrics: {
      type: 'array',
      title: 'メトリクス(JSON配列。例: ["train_error","valid_error"])',
      default: ['train_error', 'valid_error'],
    },
  },
};

const EXECUTOR_DEF_SCHEMA: JsonSchema = {
  type: 'object',
  properties: {
    network: { type: 'string', title: 'ネットワーク', default: '' },
    dataset: { type: 'string', title: 'データセット', default: '' },
  },
};

const GLOBAL_CONFIG_NAME = 'global';

/** 名前付きレコード(optimizers/monitors/executors)の追加/複製/削除+SchemaForm編集UI。 */
function NamedDefList({
  entries,
  onChange,
  schema,
  addLabel,
}: {
  entries: Record<string, Record<string, unknown>>;
  onChange: (next: Record<string, Record<string, unknown>>) => void;
  schema: JsonSchema;
  addLabel: string;
}) {
  const [newName, setNewName] = useState('');
  const names = Object.keys(entries);
  const trimmed = newName.trim();

  const handleAdd = () => {
    if (!trimmed || entries[trimmed]) return;
    onChange({ ...entries, [trimmed]: buildSchemaDefaults(schema, schema) });
    setNewName('');
  };

  const handleDuplicate = (name: string) => {
    let copyName = `${name}_copy`;
    let i = 2;
    while (entries[copyName]) {
      copyName = `${name}_copy${i}`;
      i += 1;
    }
    onChange({ ...entries, [copyName]: { ...entries[name] } });
  };

  const handleDelete = (name: string) => {
    const next = { ...entries };
    delete next[name];
    onChange(next);
  };

  return (
    <div className="grid gap-2">
      <div className="flex gap-2">
        <Input
          value={newName}
          placeholder="名前"
          onChange={(e) => setNewName(e.target.value)}
        />
        <Button
          variant="outline"
          size="sm"
          disabled={!trimmed || Boolean(entries[trimmed])}
          onClick={handleAdd}
        >
          <Plus className="size-4" />
          {addLabel}
        </Button>
      </div>
      {names.length === 0 && (
        <p className="text-[11px] text-muted-foreground">未定義</p>
      )}
      {names.map((name) => (
        <div key={name} className="grid gap-2 rounded-md border p-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold">{name}</span>
            <div className="flex gap-1">
              <Button
                variant="ghost"
                size="icon-xs"
                title="複製"
                onClick={() => handleDuplicate(name)}
              >
                <Copy className="size-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon-xs"
                title="削除"
                onClick={() => handleDelete(name)}
              >
                <Trash2 className="size-3.5" />
              </Button>
            </div>
          </div>
          <SchemaForm
            schema={schema}
            value={entries[name]}
            onChange={(v) => onChange({ ...entries, [name]: v })}
          />
        </div>
      ))}
    </div>
  );
}

export function ConfigTab() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [configSchema, setConfigSchema] = useState<JsonSchema | null>(null);
  const [newProjectName, setNewProjectName] = useState('');
  const [saveAsName, setSaveAsName] = useState('');
  const [busy, setBusy] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshProjects = async () => {
    const list = await listProjects();
    setProjects(list);
    return list;
  };

  useEffect(() => {
    // 初回マウント時のデータ取得(非同期フェッチ起点のsetStateは意図的)
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refreshProjects().catch((e) => toast.error(String(e)));
    void getConfigSchema()
      .then((s) => setConfigSchema(s as JsonSchema))
      .catch((e) => toast.error(`スキーマ取得に失敗: ${String(e)}`));
  }, []);

  useEffect(() => {
    if (!selectedId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setProject(null);
      return;
    }
    void getProject(selectedId)
      .then(setProject)
      .catch((e) => toast.error(String(e)));
  }, [selectedId]);

  // 新規プロジェクトダイアログ(TemplateGallery)でテンプレ/空プロジェクトを
  // 生成したら、CONFIGタブの一覧を再取得し当該プロジェクトを選択して反映する。
  useEffect(() => {
    const handler = (e: Event) => {
      const id = (e as CustomEvent<{ id?: string }>).detail?.id;
      if (!id) return;
      void refreshProjects()
        .then(() => setSelectedId(id))
        .catch((err) => toast.error(String(err)));
    };
    window.addEventListener(PROJECT_INSTANTIATED_EVENT, handler);
    return () => window.removeEventListener(PROJECT_INSTANTIATED_EVENT, handler);
  }, []);

  const globalConfigValue = useMemo(() => {
    if (!project || !configSchema) return {};
    return (
      (project.train_configs[GLOBAL_CONFIG_NAME] as Record<string, unknown> | undefined) ??
      buildSchemaDefaults(configSchema, configSchema)
    );
  }, [project, configSchema]);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleCreateProject = () =>
    run(async () => {
      const name = newProjectName.trim();
      if (!name) return;
      const created = await createProject(name);
      await refreshProjects();
      setSelectedId(created.id);
      setNewProjectName('');
      toast.success(`プロジェクトを作成しました: ${name}`);
    });

  const handleDeleteProject = (id: string) =>
    run(async () => {
      await deleteProject(id);
      await refreshProjects();
      if (selectedId === id) setSelectedId(null);
      toast.success('プロジェクトを削除しました');
    });

  const persist = async (patch: Parameters<typeof updateProject>[1]) => {
    if (!project) return;
    const updated = await updateProject(project.id, patch);
    setProject(updated);
    await refreshProjects();
  };

  const handleSaveGlobalConfig = (value: Record<string, unknown>) =>
    run(async () => {
      await persist({
        train_configs: { ...project?.train_configs, [GLOBAL_CONFIG_NAME]: value },
      });
      toast.success('Global Configを保存しました');
    });

  const handleSaveOptimizers = (next: Record<string, Record<string, unknown>>) =>
    run(async () => {
      await persist({ optimizers: next });
    });

  const handleSaveMonitors = (next: Record<string, Record<string, unknown>>) =>
    run(async () => {
      await persist({ monitors: next });
    });

  const handleSaveExecutors = (next: Record<string, Record<string, unknown>>) =>
    run(async () => {
      await persist({ executors: next });
    });

  // --- プロジェクト保存/開く/別名保存UX ---

  const handleDownload = () =>
    run(async () => {
      if (!project) throw new Error('プロジェクト未選択');
      const blob = new Blob([JSON.stringify(project, null, 1)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${project.name.replace(/[^0-9A-Za-z_-]+/g, '_')}.project.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success('プロジェクトJSONをダウンロードしました');
    });

  const handleUpload = (file: File) =>
    run(async () => {
      const text = await file.text();
      let doc: Partial<ProjectResponse>;
      try {
        doc = JSON.parse(text) as Partial<ProjectResponse>;
      } catch {
        throw new Error('JSONとして解釈できないファイルです');
      }
      if (typeof doc.name !== 'string') {
        throw new Error('プロジェクトJSONに name がありません');
      }
      // 2段構成: まずPOSTで新規ID/created_at/updated_atを採番し、
      // 続けてPUTでアップロードした中身(networks/train_configs等)を反映する。
      // (既存APIはPOST/GET/PUT/DELETEのみでアップロード専用エンドポイントがないため)
      const created = await createProject(doc.name, doc.description ?? '');
      const updated = await updateProject(created.id, {
        name: doc.name,
        description: doc.description ?? '',
        networks: doc.networks ?? {},
        train_configs: doc.train_configs ?? {},
        dataset_refs: doc.dataset_refs ?? [],
        run_refs: doc.run_refs ?? [],
        optimizers: doc.optimizers ?? {},
        monitors: doc.monitors ?? {},
        executors: doc.executors ?? {},
      });
      await refreshProjects();
      setSelectedId(updated.id);
      toast.success(`${file.name} を読み込みました`);
    });

  const handleSaveAs = () =>
    run(async () => {
      if (!project) throw new Error('プロジェクト未選択');
      const name = saveAsName.trim();
      if (!name) return;
      const created = await createProject(name, project.description);
      const updated = await updateProject(created.id, {
        networks: project.networks,
        train_configs: project.train_configs,
        dataset_refs: project.dataset_refs,
        run_refs: project.run_refs,
        optimizers: project.optimizers,
        monitors: project.monitors,
        executors: project.executors,
      });
      await refreshProjects();
      setSelectedId(updated.id);
      setSaveAsName('');
      toast.success(`別名で保存しました: ${name}`);
    });

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4 text-sm">
      <section className="grid gap-2">
        <div className="text-[11px] font-semibold text-muted-foreground">
          プロジェクト
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select
            value={selectedId ?? ''}
            onValueChange={(v) => setSelectedId(v || null)}
          >
            <SelectTrigger size="sm" className="w-64" data-testid="project-select">
              <SelectValue placeholder="プロジェクトを選択" />
            </SelectTrigger>
            <SelectContent>
              {projects.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {selectedId && (
            <Button
              variant="ghost"
              size="icon-xs"
              title="プロジェクトを削除"
              onClick={() => void handleDeleteProject(selectedId)}
            >
              <Trash2 className="size-3.5" />
            </Button>
          )}
        </div>

        <div className="flex gap-2">
          <Input
            value={newProjectName}
            placeholder="新規プロジェクト名"
            onChange={(e) => setNewProjectName(e.target.value)}
          />
          <Button
            variant="outline"
            size="sm"
            disabled={busy || !newProjectName.trim()}
            onClick={() => void handleCreateProject()}
          >
            <Plus className="size-4" />
            新規作成
          </Button>
        </div>

        <div className="flex flex-wrap items-center gap-2 pt-1">
          <Button
            variant="outline"
            size="sm"
            disabled={busy || !project}
            onClick={() => void handleDownload()}
          >
            <Download className="size-4" />
            ダウンロード
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleUpload(file);
              e.target.value = '';
            }}
          />
          <Button
            variant="outline"
            size="sm"
            disabled={busy}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="size-4" />
            アップロードして開く
          </Button>
        </div>

        {/*結線: エディタの複数ネットワーク(networkStore)⇔プロジェクトのnetworksマップを同期する */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <Button
            variant="outline"
            size="sm"
            disabled={busy || !project}
            onClick={() =>
              void run(async () => {
                const networks = useNetworkStore.getState().getNetworks();
                await persist({ networks });
                toast.success('エディタのネットワークをプロジェクトへ保存しました');
              })
            }
          >
            ネットワークを保存
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={busy || !project || Object.keys(project.networks ?? {}).length === 0}
            onClick={() => {
              if (!project) return;
              useNetworkStore
                .getState()
                .loadNetworks(project.networks as Record<string, IRGraphJson>);
              toast.success('プロジェクトのネットワークをエディタへ読み込みました');
            }}
          >
            ネットワークを読込
          </Button>
        </div>

        {project && (
          <div className="flex gap-2 pt-1">
            <Input
              value={saveAsName}
              placeholder="別名で保存する名前"
              onChange={(e) => setSaveAsName(e.target.value)}
            />
            <Button
              variant="outline"
              size="sm"
              disabled={busy || !saveAsName.trim()}
              onClick={() => void handleSaveAs()}
            >
              <FolderOpen className="size-4" />
              別名で保存
            </Button>
          </div>
        )}
      </section>

      {!project && (
        <p className="text-xs text-muted-foreground">
          プロジェクトを選択または新規作成してください
        </p>
      )}

      {project && configSchema && (
        <>
          <section className="grid gap-2">
            <div className="flex items-center justify-between">
              <div className="text-[11px] font-semibold text-muted-foreground">
                Global Config
              </div>
            </div>
            <SchemaForm
              schema={configSchema}
              value={globalConfigValue}
              onChange={handleSaveGlobalConfig}
            />
          </section>

          <section className="grid gap-2">
            <Label className="text-[11px] font-semibold text-muted-foreground">
              Optimizer定義
            </Label>
            <NamedDefList
              entries={
                (project.optimizers as Record<string, Record<string, unknown>>) ?? {}
              }
              onChange={handleSaveOptimizers}
              schema={OPTIMIZER_DEF_SCHEMA}
              addLabel="Optimizerを追加"
            />
          </section>

          <section className="grid gap-2">
            <Label className="text-[11px] font-semibold text-muted-foreground">
              Monitor定義
            </Label>
            <NamedDefList
              entries={
                (project.monitors as Record<string, Record<string, unknown>>) ?? {}
              }
              onChange={handleSaveMonitors}
              schema={MONITOR_DEF_SCHEMA}
              addLabel="Monitorを追加"
            />
          </section>

          <section className="grid gap-2">
            <Label className="text-[11px] font-semibold text-muted-foreground">
              Executor定義
            </Label>
            <NamedDefList
              entries={
                (project.executors as Record<string, Record<string, unknown>>) ?? {}
              }
              onChange={handleSaveExecutors}
              schema={EXECUTOR_DEF_SCHEMA}
              addLabel="Executorを追加"
            />
          </section>
        </>
      )}

      {busy && (
        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <Save className="size-3.5 animate-pulse" />
          保存中…
        </div>
      )}
    </div>
  );
}
