#include "linnet/source/source_manager.hpp"
#include "linnet/source/utf8.hpp"

#include "test.hpp"

#include <string>

using namespace linnet;

namespace {

FileId add(SourceManager& sources, std::string contents) {
    return sources.add_file("test.linnet", std::move(contents)).value();
}

} // namespace

TEST("utf8: decodes valid sequences of every length") {
    CHECK_EQ(decode_utf8("a", 0).value, U'a');
    const std::string text = "\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80"; // é € 😀
    const DecodedCodePoint two = decode_utf8(text, 0);
    CHECK(two.valid && two.value == 0xE9 && two.length == 2);
    const DecodedCodePoint three = decode_utf8(text, 2);
    CHECK(three.valid && three.value == 0x20AC && three.length == 3);
    const DecodedCodePoint four = decode_utf8(text, 5);
    CHECK(four.valid && four.value == 0x1F600 && four.length == 4);
}

TEST("utf8: malformed input consumes one byte") {
    const std::string cases[] = {
        "\x80",             // stray continuation
        "\xC3",             // truncated two-byte sequence
        "\xE2\x82",         // truncated three-byte sequence
        "\xC0\xAF",         // overlong
        "\xE0\x80\xAF",     // overlong
        "\xED\xA0\x80",     // surrogate
        "\xF4\x90\x80\x80", // above U+10FFFF
        "\xFF",             // invalid lead byte
        "\xC3\x28",         // bad continuation
    };
    for (const std::string& text : cases) {
        const DecodedCodePoint decoded = decode_utf8(text, 0);
        CHECK(!decoded.valid);
        CHECK_EQ(decoded.length, 1U);
        CHECK(decoded.value == replacement_character);
    }
}

TEST("source manager: empty file has one empty line") {
    SourceManager sources;
    const FileId file = add(sources, "");
    CHECK_EQ(sources.line_count(file), 1U);
    CHECK_EQ(sources.line_text(file, 1), "");
    CHECK(sources.line_column(file, 0) == LineColumn{1, 1});
    CHECK(sources.line_column(file, 99) == LineColumn{1, 1});
}

TEST("source manager: maps offsets to lines and columns") {
    SourceManager sources;
    const FileId file = add(sources, "ab\ncd\r\n\nxyz");
    CHECK_EQ(sources.line_count(file), 4U);
    CHECK_EQ(sources.line_text(file, 1), "ab");
    CHECK_EQ(sources.line_text(file, 2), "cd");
    CHECK_EQ(sources.line_text(file, 3), "");
    CHECK_EQ(sources.line_text(file, 4), "xyz");
    CHECK_EQ(sources.line_start(file, 4), 8U);

    CHECK(sources.line_column(file, 0) == LineColumn{1, 1});
    CHECK(sources.line_column(file, 2) == LineColumn{1, 3}); // at '\n'
    CHECK(sources.line_column(file, 3) == LineColumn{2, 1});
    CHECK(sources.line_column(file, 7) == LineColumn{3, 1});
    CHECK(sources.line_column(file, 11) == LineColumn{4, 4}); // end of file
}

TEST("source manager: trailing newline yields a final empty line") {
    SourceManager sources;
    const FileId file = add(sources, "a\n");
    CHECK_EQ(sources.line_count(file), 2U);
    CHECK_EQ(sources.line_text(file, 2), "");
    CHECK(sources.line_column(file, 2) == LineColumn{2, 1});
}

TEST("source manager: span text") {
    SourceManager sources;
    const FileId file = add(sources, "let x = 1");
    CHECK_EQ(sources.text({file, 4, 5}), "x");
    CHECK_EQ(sources.text({file, 9, 9}), "");
}

TEST("source manager: views stay valid as files are added") {
    SourceManager sources;
    const FileId first = add(sources, "short");
    const std::string_view view = sources.contents(first);
    for (int i = 0; i < 100; ++i) {
        add(sources, "filler");
    }
    CHECK_EQ(view, "short");
    CHECK(view.data() == sources.contents(first).data());
}

TEST("source manager: UTF-16 positions") {
    SourceManager sources;
    // line 0: a é 😀 b   (bytes: 1 + 2 + 4 + 1)
    const FileId file = add(sources,
                            "a\xC3\xA9\xF0\x9F\x98\x80"
                            "b\nz");
    CHECK(sources.utf16_position(file, 0) == Utf16Position{0, 0});
    CHECK(sources.utf16_position(file, 1) == Utf16Position{0, 1});
    CHECK(sources.utf16_position(file, 3) == Utf16Position{0, 2});
    CHECK(sources.utf16_position(file, 7) == Utf16Position{0, 4});
    CHECK(sources.utf16_position(file, 8) == Utf16Position{0, 5});
    CHECK(sources.utf16_position(file, 9) == Utf16Position{1, 0});
    // Offsets inside a code point report the start of that code point.
    CHECK(sources.utf16_position(file, 5) == Utf16Position{0, 2});

    CHECK_EQ(sources.offset_from_utf16(file, {0, 0}), 0U);
    CHECK_EQ(sources.offset_from_utf16(file, {0, 2}), 3U);
    CHECK_EQ(sources.offset_from_utf16(file, {0, 3}), 3U); // inside surrogate pair
    CHECK_EQ(sources.offset_from_utf16(file, {0, 4}), 7U);
    CHECK_EQ(sources.offset_from_utf16(file, {0, 99}), 8U); // clamps to line end
    CHECK_EQ(sources.offset_from_utf16(file, {1, 1}), 10U);
    CHECK_EQ(sources.offset_from_utf16(file, {7, 0}), 10U); // clamps to file end
}

TEST("source manager: UTF-16 conversion tolerates invalid UTF-8") {
    SourceManager sources;
    const FileId file = add(sources, "\xFF\xC3z");
    CHECK(sources.utf16_position(file, 3) == Utf16Position{0, 3});
    CHECK_EQ(sources.offset_from_utf16(file, {0, 2}), 2U);
}

TEST("source manager: load_file reports missing files") {
    SourceManager sources;
    const auto result = sources.load_file("does/not/exist.linnet");
    CHECK(!result.has_value());
    CHECK(result.error().find("does/not/exist.linnet") != std::string::npos);
}
