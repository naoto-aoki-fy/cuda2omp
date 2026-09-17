// Native, source-location-safe parsing and rewrite foundation for cuda2omp.
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <memory>
#include <string>

using namespace clang;
using namespace clang::tooling;

namespace {
llvm::cl::OptionCategory Category("cuda2omp native frontend options");
llvm::cl::opt<std::string> Output("o", llvm::cl::desc("Output main-file source"),
                                  llvm::cl::Required, llvm::cl::cat(Category));
llvm::cl::list<std::string> Renames(
    "rename", llvm::cl::desc("Rename a resolved reference (OLD=NEW)"),
    llvm::cl::ZeroOrMore, llvm::cl::cat(Category));

class Visitor final : public RecursiveASTVisitor<Visitor> {
public:
  Visitor(ASTContext &Context, Rewriter &Rewrite,
          const std::map<std::string, std::string> &RenameMap)
      : Context(Context), Rewrite(Rewrite), SM(Context.getSourceManager()),
        RenameMap(RenameMap) {}

  bool VisitFunctionDecl(FunctionDecl *Declaration) {
    if (!Declaration->hasAttr<CUDADeviceAttr>() &&
        !Declaration->hasAttr<CUDAGlobalAttr>())
      return true;
    // CUDA declarations are transformation candidates even before a lowering
    // is implemented. Validate their spelling location now, so a declaration
    // in an include can never accidentally become a numeric-offset edit.
    validate(Declaration->getLocation(), "CUDA declaration");
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *Reference) {
    const auto Found = RenameMap.find(Reference->getNameInfo().getAsString());
    if (Found == RenameMap.end())
      return true;
    SourceLocation Location = Reference->getLocation();
    if (!validate(Location, "reference"))
      return true;
    const CharSourceRange Token = CharSourceRange::getTokenRange(Location, Location);
    if (Rewrite.ReplaceText(Token, Found->second))
      diagnose(Location, "cuda2omp: failed to rewrite candidate reference");
    return true;
  }

private:
  bool validate(SourceLocation Location, StringRef Kind) {
    if (Location.isInvalid()) {
      diagnose(Location, "cuda2omp: candidate has no valid source location");
      return false;
    }
    if (Location.isMacroID()) {
      diagnose(Location,
               "cuda2omp: non-rewritable candidate in macro expansion");
      return false;
    }
    const SourceLocation Spelling = SM.getSpellingLoc(Location);
    if (Spelling.isInvalid() || !SM.isWrittenInMainFile(Spelling)) {
      const std::string Message =
          ("cuda2omp: " + Kind + " is declared in an included file; not editing")
              .str();
      diagnose(Location, Message);
      return false;
    }
    if (!Rewriter::isRewritable(Spelling)) {
      diagnose(Location, "cuda2omp: candidate source range is not rewritable");
      return false;
    }
    return true;
  }

  void diagnose(SourceLocation Location, StringRef Message) {
    DiagnosticsEngine &Diagnostics = Context.getDiagnostics();
    const unsigned ID = Diagnostics.getCustomDiagID(DiagnosticsEngine::Warning,
                                                     Message);
    Diagnostics.Report(Location, ID);
  }

  ASTContext &Context;
  Rewriter &Rewrite;
  SourceManager &SM;
  const std::map<std::string, std::string> &RenameMap;
};

class Consumer final : public ASTConsumer {
public:
  Consumer(ASTContext &Context, Rewriter &Rewrite,
           const std::map<std::string, std::string> &RenameMap)
      : ASTVisitor(Context, Rewrite, RenameMap) {}
  void HandleTranslationUnit(ASTContext &Context) override {
    ASTVisitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  Visitor ASTVisitor;
};

class Action final : public ASTFrontendAction {
public:
  explicit Action(std::map<std::string, std::string> RenameMap)
      : RenameMap(std::move(RenameMap)) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler, StringRef) override {
    Rewrite.setSourceMgr(Compiler.getSourceManager(), Compiler.getLangOpts());
    return std::make_unique<Consumer>(Compiler.getASTContext(), Rewrite,
                                      RenameMap);
  }

  void EndSourceFileAction() override {
    const SourceManager &SM = Rewrite.getSourceMgr();
    const FileID Main = SM.getMainFileID();
    std::error_code Error;
    llvm::raw_fd_ostream Stream(Output, Error);
    if (Error) {
      llvm::errs() << "cuda2omp: cannot open " << Output << ": "
                   << Error.message() << '\n';
      return;
    }
    if (const RewriteBuffer *Buffer = Rewrite.getRewriteBufferFor(Main))
      Buffer->write(Stream);
    else
      Stream << SM.getBufferData(Main);
  }

private:
  Rewriter Rewrite;
  std::map<std::string, std::string> RenameMap;
};

class Factory final : public FrontendActionFactory {
public:
  explicit Factory(std::map<std::string, std::string> RenameMap)
      : RenameMap(std::move(RenameMap)) {}
  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<Action>(RenameMap);
  }

private:
  std::map<std::string, std::string> RenameMap;
};
} // namespace

int main(int argc, const char **argv) {
  auto Parser = CommonOptionsParser::create(argc, argv, Category);
  if (!Parser) {
    llvm::errs() << llvm::toString(Parser.takeError()) << '\n';
    return 1;
  }
  if (Parser->getSourcePathList().size() != 1) {
    llvm::errs() << "cuda2omp: exactly one input translation unit is required\n";
    return 1;
  }
  std::map<std::string, std::string> RenameMap;
  for (const std::string &Rule : Renames) {
    const size_t Equal = Rule.find('=');
    if (Equal == std::string::npos || Equal == 0 || Equal + 1 == Rule.size()) {
      llvm::errs() << "cuda2omp: --rename expects OLD=NEW\n";
      return 1;
    }
    RenameMap.emplace(Rule.substr(0, Equal), Rule.substr(Equal + 1));
  }
  ClangTool Tool(Parser->getCompilations(), Parser->getSourcePathList());
  Factory Actions(std::move(RenameMap));
  return Tool.run(&Actions);
}
