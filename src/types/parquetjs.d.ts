declare module "parquetjs" {
  export class ParquetReader {
    static openFile(filename: string): Promise<ParquetReader>;
    getCursor(): ParquetCursor;
    close(): Promise<void>;
  }

  export class ParquetCursor {
    next(): Promise<Record<string, unknown> | null>;
  }
}
