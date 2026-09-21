#pragma once

#include <cstdint>
#include <deque>
#include <expected>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace linnet {

using FileId = std::uint32_t;

inline constexpr FileId invalid_file_id = std::numeric_limits<FileId>::max();

// Half-open byte range [begin, end) within one file.
struct SourceSpan {
    FileId file = invalid_file_id;
    std::uint32_t begin = 0;
    std::uint32_t end = 0;

    constexpr std::uint32_t size() const { return end - begin; }
    constexpr bool empty() const { return begin == end; }
    friend constexpr bool operator==(const SourceSpan&, const SourceSpan&) = default;
};

// Human-facing position: 1-based line, 1-based column counted in bytes.
struct LineColumn {
    std::uint32_t line = 1;
    std::uint32_t column = 1;
    friend constexpr bool operator==(const LineColumn&, const LineColumn&) = default;
};

// Language-server position: 0-based line, 0-based offset in UTF-16 code units.
struct Utf16Position {
    std::uint32_t line = 0;
    std::uint32_t character = 0;
    friend constexpr bool operator==(const Utf16Position&, const Utf16Position&) = default;
};

// Owns immutable file contents and maps byte offsets to positions. Offsets are
// bytes everywhere; UTF-16 conversion exists only for the LSP boundary.
//
// Lines are terminated by "\n"; a preceding "\r" belongs to the terminator.
// Contents are never modified after registration, so the views returned here
// stay valid for the lifetime of the manager. An edited document is registered
// as a new file.
class SourceManager {
public:
    // Largest accepted file: offsets must fit in 32 bits.
    static constexpr std::size_t max_file_size = std::numeric_limits<std::uint32_t>::max() - 1;

    // Registers in-memory contents under a display path.
    std::expected<FileId, std::string> add_file(std::string path, std::string contents);

    // Reads a file from disk as raw bytes; no encoding validation is performed.
    std::expected<FileId, std::string> load_file(const std::filesystem::path& path);

    std::size_t file_count() const { return files_.size(); }
    std::string_view path(FileId file) const;
    std::string_view contents(FileId file) const;
    std::uint32_t size(FileId file) const;
    std::string_view text(SourceSpan span) const;

    // Number of lines; an empty file and a file without a trailing newline
    // both end in one (possibly empty) final line.
    std::uint32_t line_count(FileId file) const;

    // Text of a 1-based line, without its terminator.
    std::string_view line_text(FileId file, std::uint32_t line) const;

    // Byte offset of the start of a 1-based line.
    std::uint32_t line_start(FileId file, std::uint32_t line) const;

    // Offsets past the end of the file are clamped to the end.
    LineColumn line_column(FileId file, std::uint32_t offset) const;
    Utf16Position utf16_position(FileId file, std::uint32_t offset) const;

    // Inverse of utf16_position. Positions beyond the end of a line clamp to
    // the end of that line; lines beyond the end of the file clamp to the end
    // of the file. A position inside a surrogate pair maps to the start of
    // that code point.
    std::uint32_t offset_from_utf16(FileId file, Utf16Position position) const;

private:
    struct File {
        std::string path;
        std::string contents;
        std::vector<std::uint32_t> line_starts;
    };

    const File& file(FileId id) const;
    static std::uint32_t line_index(const File& file, std::uint32_t offset);
    static std::uint32_t line_end(const File& file, std::uint32_t line_index);

    // deque keeps element addresses stable as files are added.
    std::deque<File> files_;
};

} // namespace linnet
